#!/usr/bin/env sh
set -e
SFD="webapp/DebruitsRegular-Handwritten.sfd"
TTF="webapp/DebruitsRegular-Handwritten.ttf"
fontforge -lang=ff -c "Open(\"$SFD\"); Generate(\"$TTF\");"
echo "Generated $TTF"
