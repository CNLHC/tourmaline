#!/usr/bin/env python3

import os
import re
import string
import subprocess
import sys
import shutil
import unidecode
import argparse
import logging


from argparse import FileType

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTChar, LTFigure, LTTextBox, LTTextLine

logger = logging.getLogger(__name__)

__all__ = ['pdf_title']


def make_parsing_state(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('ParsingState', (), enums)


CHAR_PARSING_STATE = make_parsing_state('INIT_X', 'INIT_D', 'INSIDE_WORD')
MIN_CHARS = 6
MAX_WORDS = 20
MAX_CHARS = MAX_WORDS * 10
TOLERANCE = 1e-06
IS_LOG_ON = False


def sanitize(filename):
    """Turn string into a valid file name.
    """
    # If the title was picked up from text, it may be too large.
    # Preserve a certain number of words and characters
    words = filename.split(' ')
    filename = ' '.join(words[0:MAX_WORDS])
    if len(filename) > MAX_CHARS:
        filename = filename[0:MAX_CHARS]

    # Preserve letters with diacritics
    try:
        filename = unidecode.unidecode(
            filename.encode('utf-8').decode('utf-8'))
    except UnicodeDecodeError:
        logger.info(
            "*** Skipping invalid title decoding for file %s! ***" % filename)

    # Preserve subtitle and itemization separators
    filename = re.sub(r',', ' ', filename)
    filename = re.sub(r': ', ' - ', filename)

    # Strip repetitions
    filename = re.sub(r'\.pdf(\.pdf)*$', '', filename)
    filename = re.sub(r'[ \t][ \t]*', ' ', filename)

    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join([c for c in filename if c in valid_chars])


def meta_title(filename):
    """Title from pdf metadata.
    """
    docinfo = PDFDocument(PDFParser(open(filename, 'rb'))).info
    if docinfo is None:
        return ''
    for meta in docinfo:
        if hasattr(meta, 'title'):
            return meta['title']
        elif hasattr(meta, 'Title'):
            return meta['Title']
        elif hasattr(meta, 'TITLE'):
            return meta['TITLE']
    return ''


def junk_line(line):
    """Judge if a line is not appropriate for a title.
    """
    too_small = len(line.strip()) < MIN_CHARS
    is_placeholder_text = bool(re.search(
        r'^[0-9 \t-]+(abstract|introduction)?\s+$|^(abstract|unknown|title|untitled):?$', line.strip().lower()))
    is_copyright_info = bool(re.search(
        r'paper\s+title|technical\s+report|proceedings|preprint|to\s+appear|submission|(integrated|international).*conference|transactions\s+on|symposium\s+on|downloaded\s+from\s+http', line.lower()))

    # NOTE: Titles which only contain a number will be discarded
    stripped_to_ascii = ''.join(
        [c for c in line.strip() if c in string.ascii_letters])
    ascii_length = len(stripped_to_ascii)
    stripped_to_chars = re.sub(r'[ \t\n]', '', line.strip())
    chars_length = len(stripped_to_chars)
    is_serial_number = ascii_length < chars_length / 2

    return too_small or is_placeholder_text or is_copyright_info or is_serial_number


def empty_str(s):
    return len(s.strip()) == 0


def is_close(a, b, relative_tolerance=TOLERANCE):
    return abs(a-b) <= relative_tolerance * max(abs(a), abs(b))


def update_largest_text(line, y0, size, largest_text):
    logger.debug('update size: ' + str(size))
    logger.debug('largest_text size: ' + str(largest_text['size']))

    # Sometimes font size is not correctly read, so we
    # fallback to text y0 (not even height may be calculated).
    # In this case, we consider the first line of text to be a title.
    if ((size == largest_text['size'] == 0) and (y0 - largest_text['y0'] < -TOLERANCE)):
        return largest_text

    # If it is a split line, it may contain a new line at the end
    line = re.sub(r'\n$', ' ', line)

    if (size - largest_text['size'] > TOLERANCE):
        largest_text = {
            'contents': line,
            'y0': y0,
            'size': size
        }
    # Title spans multiple lines
    elif is_close(size, largest_text['size']):
        largest_text['contents'] = largest_text['contents'] + line
        largest_text['y0'] = y0

    return largest_text


def extract_largest_text(obj, largest_text):
    # Skip first letter of line when calculating size, as articles
    # may enlarge it enough to be bigger then the title size.
    # Also skip other elements such as `LTAnno`.
    for i, child in enumerate(obj):
        if isinstance(child, LTTextLine):
            logger.debug('lt_obj child line: ' + str(child))
            for j, child2 in enumerate(child):
                if j > 1 and isinstance(child2, LTChar):
                    largest_text = update_largest_text(
                        child.get_text(), child2.y0, child2.size, largest_text)
                    # Only need to parse size of one char
                    break
        elif i > 1 and isinstance(child, LTChar):
            logger.debug('lt_obj child char: ' + str(child))
            largest_text = update_largest_text(
                obj.get_text(), child.y0, child.size, largest_text)
            # Only need to parse size of one char
            break
    return largest_text


def extract_figure_text(lt_obj, largest_text):
    """
    Extract text contained in a `LTFigure`.
    Since text is encoded in `LTChar` elements, we detect separate lines
    by keeping track of changes in font size.
    """
    text = ''
    line = ''
    y0 = 0
    size = 0
    char_distance = 0
    char_previous_x1 = 0
    state = CHAR_PARSING_STATE.INIT_X
    for child in lt_obj:
        logger.debug('child: ' + str(child))

        # Ignore other elements
        if not isinstance(child, LTChar):
            continue

        char_y0 = child.y0
        char_size = child.size
        char_text = child.get_text()
        decoded_char_text = unidecode.unidecode(
            char_text.encode('utf-8').decode('utf-8'))
        logger.debug('char: ' + str(char_size) + ' ' + str(decoded_char_text))

        # A new line was detected
        if char_size != size:
            logger.debug('new line')
            largest_text = update_largest_text(line, y0, size, largest_text)
            text += line + '\n'
            line = char_text
            y0 = char_y0
            size = char_size

            char_previous_x1 = child.x1
            state = CHAR_PARSING_STATE.INIT_D
        else:
            # Spaces may not be present as `LTChar` elements,
            # so we manually add them.
            # NOTE: A word starting with lowercase can't be
            # distinguished from the current word.
            char_current_distance = abs(child.x0 - char_previous_x1)
            logger.debug('char_current_distance: ' +
                         str(char_current_distance))
            logger.debug('char_distance: ' + str(char_distance))
            logger.debug('state: ' + str(state))

            # Initialization
            if state == CHAR_PARSING_STATE.INIT_X:
                char_previous_x1 = child.x1
                state = CHAR_PARSING_STATE.INIT_D
            elif state == CHAR_PARSING_STATE.INIT_D:
                # Update distance only if no space is detected
                if (char_distance > 0) and (char_current_distance < char_distance * 2.5):
                    char_distance = char_current_distance
                if (char_distance < 0.1):
                    char_distance = 0.1
                state = CHAR_PARSING_STATE.INSIDE_WORD
            # If the x-position decreased, then it's a new line
            if (state == CHAR_PARSING_STATE.INSIDE_WORD) and (child.x1 < char_previous_x1):
                logger.debug('x-position decreased')
                line += ' '
                char_previous_x1 = child.x1
                state = CHAR_PARSING_STATE.INIT_D
            # Large enough distance: it's a space
            elif (state == CHAR_PARSING_STATE.INSIDE_WORD) and (char_current_distance > char_distance * 8.5):
                logger.debug('space detected')
                logger.debug('char_current_distance: ' +
                             str(char_current_distance))
                logger.debug('char_distance: ' + str(char_distance))
                line += ' '
                char_previous_x1 = child.x1
            # When larger distance is detected between chars, use it to
            # improve our heuristic
            elif (state == CHAR_PARSING_STATE.INSIDE_WORD) and (char_current_distance > char_distance) and (char_current_distance < char_distance * 2.5):
                char_distance = char_current_distance
                char_previous_x1 = child.x1
            # Chars are sequential
            else:
                char_previous_x1 = child.x1
            child_text = child.get_text()
            if not empty_str(child_text):
                line += child_text
    return (largest_text, text)


def pdf_text(filename):
    fp = open(filename, 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser, '')
    parser.set_document(doc)
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    text = ''
    largest_text = {
        'contents': '',
        'y0': 0,
        'size': 0
    }
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
        layout = device.get_result()
        for lt_obj in layout:
            logger.debug('lt_obj: ' + str(lt_obj))
            if isinstance(lt_obj, LTFigure):
                (largest_text, figure_text) = extract_figure_text(
                    lt_obj, largest_text)
                text += figure_text
            elif isinstance(lt_obj, (LTTextBox, LTTextLine)):
                # Ignore body text blocks
                stripped_to_chars = re.sub(
                    r'[ \t\n]', '', lt_obj.get_text().strip())
                if (len(stripped_to_chars) > MAX_CHARS * 2):
                    continue

                largest_text = extract_largest_text(lt_obj, largest_text)
                text += lt_obj.get_text() + '\n'

        # Remove unprocessed CID text
        largest_text['contents'] = re.sub(
            r'(\(cid:[0-9 \t-]*\))*', '', largest_text['contents'])

        # Only parse the first page
        return (largest_text, text)


def title_start(lines):
    for i, line in enumerate(lines):
        if not empty_str(line) and not junk_line(line):
            return i
    return 0


def title_end(lines, start, max_lines=2):
    for i, line in enumerate(lines[start+1:start+max_lines+1], start+1):
        if empty_str(line):
            return i
    return start + 1


def text_title(filename):
    """Extract title from PDF's text.
    """
    (largest_text, lines_joined) = pdf_text(filename)

    if empty_str(largest_text['contents']):
        lines = lines_joined.strip().split('\n')
        i = title_start(lines)
        j = title_end(lines, i)
        text = ' '.join(line.strip() for line in lines[i:j])
    else:
        text = largest_text['contents'].strip()

    # Strip dots, which conflict with os.path's splittext()
    text = re.sub(r'\.', '', text)

    # Strip extra whitespace
    text = re.sub(r'[\t\n]', '', text)

    return text


def pdftotext_title(filename):
    """Extract title using `pdftotext`
    """
    command = 'pdftotext {} -'.format(re.sub(' ', '\\ ', filename))
    process = subprocess.Popen([command],
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out, err = process.communicate()
    lines = out.strip().split('\n')

    i = title_start(lines)
    j = title_end(lines, i)
    text = ' '.join(line.strip() for line in lines[i:j])

    # Strip dots, which conflict with os.path's splittext()
    text = re.sub(r'\.', '', text)

    # Strip extra whitespace
    text = re.sub(r'[\t\n]', '', text)

    return text


def valid_title(title):
    return not empty_str(title) and not junk_line(title) and empty_str(os.path.splitext(title)[1])


def pdf_title(filename):
    """Extract title using one of multiple strategies.
    """
    try:
        title = meta_title(filename)
        if valid_title(title):
            return title
    except Exception as e:
        logger.info("*** Skipping invalid metadata for file %s! ***" % filename)

    try:
        title = text_title(filename)
        if valid_title(title):
            return title
    except Exception as e:
        logger.info("*** Skipping invalid parsing for file %s! ***" % filename)

    title = pdftotext_title(filename)
    if valid_title(title):
        return title

    return os.path.basename(os.path.splitext(filename)[0])


def DirPath(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


parser = argparse.ArgumentParser(description='Extract title from PDF file.')
parser.add_argument('--dist', dest="dist", type=DirPath, default='.')
parser.add_argument('--override', dest='override', action='store_true')
parser.add_argument('--rename', dest='rename', action='store_true')
parser.add_argument('--dry-run', dest="dryRun", action='store_true')
parser.add_argument('--underscore', dest="underscore", action='store_true')
parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
parser.add_argument('-vvv', '--debug', dest="debug", action='store_true')
parser.add_argument('filenames', nargs="+")

fh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(levelname)8s]: %(message)s')
fh.setFormatter(formatter)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        # logging.basicConfig(level=logging.INFO)
        fh.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    if args.debug:
        fh.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.addHandler(fh)

    for filename in args.filenames:
        title = pdf_title(filename)
        title = sanitize(' '.join(title.split()))
        if args.underscore:
            title = title.replace(" ", "_")

        if args.rename:
            new_name = os.path.join(args.dist, title + ".pdf")
            logger.warning("Rename: %s => %s" % (filename, new_name))
            if not args.dryRun:
                if os.path.exists(new_name):
                    if not args.override:
                        logger.error("Target %s already exists! " % new_name)
                        sys.exit(-1)
                    else:
                        logger.warning("Override %s" % new_name)
                        shutil.copy(filename, new_name)
                else:
                    shutil.copy(filename, new_name)
        else:
            sys.stdout.write(title)
