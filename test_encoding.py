import sys
import codecs
if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
print('「')
