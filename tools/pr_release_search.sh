#!/bin/bash

# Search for merged pull requests since the last tag for a given label.
#
# Examples:
#
#   Find merged pull requests for essreduce since 26.6.0 release:
#     pr_release_search.sh essreduce/26.6.0
#
#   Find merged pull requests for essreduce since latest release of that package:
#     pr_release_search.sh essreduce

TAG="$1"


LABEL="${TAG%%/*}"
VERSION="${TAG#*/}"


# If version == label it means no version was provided, so we should use the latest tag
# that contains the label as a prefix.
if [ "$VERSION" == "$LABEL" ]; then

  TAG=$(git describe --tags --abbrev=0 --match "$LABEL/*")
  echo "Using tag: $TAG"
fi


SINCE=$(git log -1 --format=%cs "$TAG")

gh pr list \
  --state merged \
  --label "$LABEL" \
  --search "merged:>=$SINCE" \
  --limit 500
