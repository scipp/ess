#!bin/bash

CUR_TAG=$(git describe --tags --abbrev=0)
# Prepare Cache Folder
GIT_TOP_DIR=$(git rev-parse --show-toplevel)
CACHE_DIR=${GIT_TOP_DIR}/.ess_release_cache

RELEASE_COMMITS_LIST=${CACHE_DIR}/${CUR_TAG/\//-}-commits.txt

echo "Current tag: ${CUR_TAG}";

PACKAGE=""
VERSION=""

if [[ "${CUR_TAG}" =~ ^(^ess[^@]+)\/([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
  PACKAGE=${BASH_REMATCH[1]};
  VERSION=${BASH_REMATCH[2]};
else
  echo "Tag name unexpected...";
  exit 1;
fi

echo "Package: ${PACKAGE}";
echo "Version: ${VERSION}";

find_last_release() {
  previous_releases=($(git tag -l ${PACKAGE}* --sort creatordate));
  echo $previous_releases
  if [[ ${#previous_releases[@]} -gt 1 ]]; then
    LAST_RELEASE=${previous_releases[-2]};
  else
	echo "No previous release found on the package yet. Can't collect release notes automatically..."
	exit 1;
  fi
}

find_last_release
echo "Found the previous release on the package ${PACKAGE}: ${LAST_RELEASE}";

commits_between=($(git log --first-parent --merges --pretty=format:%H ${LAST_RELEASE}...${CUR_TAG}));
echo ${commits_between} >> ${RELEASE_COMMITS_LIST};

download_commit_and_pr_info () {
  for commit in ${commits_between[@]}; do
    commit_dir=${CACHE_DIR}/${commit}
    if [ -d ${commit_dir} ]; then
	  echo "commit cache directory already exists with these files: $(ls ${commit_dir})";
    else
	  mkdir --parent ${commit_dir};
    fi

    pr_json=${commit_dir}/all-prs.json;
    if [ -f ${pr_json} ]; then
	  echo "${pr_json} already exists. skipping downloading..."
    else
      echo $(gh api \
        -H "Accept: application/vnd.github+json" \
		-H "X-GitHub-Api-Version: 2026-03-10" \
    	/repos/scipp/ess/commits/${commit}/pulls) >> ${pr_json};
	  echo "PR information associated with ${commit} downloaded to ${pr_json}";
    fi
    pr_numbers=($(jq .[].number ${pr_json}));
	for pr_number in ${pr_numbers[@]}; do
	  echo ${pr_number};
	  pr_commit_json=${commit_dir}/pr-${pr_number}-commits.json;
	  if [ -f ${pr_commit_json} ]; then
		echo "${pr_commit_json} already exists. Skipping downloading...";
	  else
		echo $(gh api \
		  -H "Accept: application/vnd.github+json" \
		  -H "X-GitHub-Api-Version: 2026-03-10" \
		  /repos/scipp/ess/pulls/${pr_number}/commits) >> ${pr_commit_json};
	  fi
    done
  done
}

download_commit_and_pr_info;

