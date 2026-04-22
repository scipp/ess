import argparse
import json
import pathlib

from pydantic import BaseModel, ConfigDict

ROOT_DIR = pathlib.Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / ".ess_release_cache"


class Label(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)
    name: str


class GHUser(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)
    login: str


class PRDescription(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)
    title: str
    url: str
    number: int
    user: GHUser
    issue_url: str
    labels: list[Label]


class CommitDescription(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)
    author: GHUser


class MergeLog(BaseModel):
    authors: list[GHUser]
    pr: PRDescription

    def __str__(self) -> str:
        authors = ", ".join([f"@{author.login}" for author in self.authors])
        return f"{self.pr.title} by {authors} in {self.pr.url}"


def get_commits_file(cur_tag: str, compare_tag: str) -> pathlib.Path:
    file_path = CACHE_DIR / pathlib.Path(
        f"{cur_tag.replace('/', '-')}-{compare_tag.replace('/', '-')}-commits.txt"
    )
    if not file_path.exists():
        raise ValueError(
            f"{file_path.as_posix()} doesn't exist. "
            "Download/save git commit and PR history first."
        )
    return file_path


def has_label(label: str, pr: PRDescription) -> bool:
    return any(label.lower() == pr_label.name.lower() for pr_label in pr.labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cur-tag", help="Current Tag for Release")
    parser.add_argument("--compare-tag", help="Last Tag for Comparison")
    parser.add_argument("--output-file-path", help="Path to save the output.")

    args = parser.parse_args()
    cur_tag = args.cur_tag
    compare_tag = args.compare_tag
    package_name = cur_tag.split('/')[0]
    commit_list_file = get_commits_file(cur_tag, compare_tag)

    commits = commit_list_file.read_text().splitlines()
    merge_logs = []
    for commit in commits:
        pr_json = CACHE_DIR / commit / "all-prs.json"
        pr_list = json.loads(pr_json.read_text())
        for pr in pr_list:
            pr_obj = PRDescription(**pr)
            pr_commits_list_json = (
                CACHE_DIR / commit / f"pr-{pr_obj.number}-commits.json"
            )
            pr_commits_list = json.loads(pr_commits_list_json.read_text())
            authors = set()
            for pr_commit in pr_commits_list:
                pr_commit_obj = CommitDescription(**pr_commit)
                authors.add(pr_commit_obj.author)
            merge_logs.append(MergeLog(pr=pr_obj, authors=list(authors)))

    relevant_logs = []
    maybe_relevant_logs = []
    for merge_log in merge_logs:
        if has_label(package_name, merge_log.pr):
            relevant_logs.append(merge_log)
        else:
            maybe_relevant_logs.append(merge_log)

    release_note = [
        "## What's Changed",
        '',
        f"### {package_name.replace('ess', 'ESS')}",
        '',
    ]
    release_note.extend([f"* {relevant_log}" for relevant_log in relevant_logs])

    release_note.extend(['', '### Maybe Related Changes', ''])
    release_note.extend([f"* {relevant_log}" for relevant_log in maybe_relevant_logs])
    release_note.extend(
        [
            '',
            '',
            f"**Full Changelog**: https://github.com/scipp/ess/compare/{compare_tag}...{cur_tag}",
            '',
        ]
    )

    if hasattr(args, 'output_file_path') and args.output_file_path:
        output_file_path = pathlib.Path(args.output_file_path)
    else:
        output_file_path = CACHE_DIR / commit_list_file.name.replace(
            '-commits.txt', '-releasenote.md'
        )
    output_file_path.write_text('\n'.join(release_note) + '\n')
