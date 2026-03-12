#!/bin/bash

RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}Repository name${GREEN} to move issues from:${NC}"
read  repo_name
echo -e "Transferring issues from ${GREEN}${repo_name}${NC} to the new ${RED}``ess``${NC} repository..."

REPO_FLAG="-R scipp/${repo_name}"
numbers=$(gh issue list ${REPO_FLAG} --limit 200 --state all --json number -q ".[].number")

labels_str=$(gh label list ${REPO_FLAG} --json name -q ".[].name")
labels=($labels_str)

label_exist=0
for label in "${labels[@]}"; do
    if [[ $label = "${repo_name}" ]]; then
        label_exist=1;
        break;
    fi
done
if [[ $label_exist = 0 ]]; then
    echo -e "Label ${repo_name} does not exist. Creating one...";
    gh label create ${repo_name} ${REPO_FLAG} --description "Issues for ${repo_name}.";
    # Creating the same label so that it will be kept after transfer.
    gh label create ${repo_name} -R scipp/ess --description "Issues for ${repo_name}.";
else
    echo -e "Label ${repo_name} exists. Assigning the labels to the issues to be moved...";
fi

for number in ${numbers};
do
	read -r -a issue_labels <<< $(gh issue view ${number} --json labels -q ".labels[].name" ${REPO_FLAG});
	issue_label_exist=0
        for issue_label in "${issue_labels[@]}"; do
	  if [[ ${issue_label} = ${repo_name} ]]; then
	    issue_label_exist=1;
	    break;
	  fi
	done

	if [[ ${issue_label_exist} = 0 ]]; then
	  echo -e "Adding label ${BLUE}${repo_name}${NC} to issue #${GREEN}${number}${NC} ...";
	  gh issue edit ${number} --add-label ${repo_name} ${REPO_FLAG};
        else
	  echo -e "Label ${BLUE}${repo_name}${NC} already assigned to issues #${GREEN}${number}${NC}.";
	fi

done

num_issues=${#numbers[@]};
if [[ ${num_issues} = 1 ]]; then
  if [[ ${#numbers[0]} = 0 ]]; then
	  unset numbers[0];
  fi
  num_issues=${#numbers[@]};
fi

if [[ ${num_issues} = 200 ]]; then
	echo -e "Found ${RED}${num_issue}${NC} issues. There may be more issues left.";
fi

echo "Transferring ${num_issues} issues..."
for number in ${numbers};
do
	echo -e "Transferring issue #${GREEN}${number}${NC} from ${BLUE}${repo_name}${NC} to ${RED}ess${NC} repository...";
	gh issue transfer ${number} scipp/ess ${REPO_FLAG};
done

