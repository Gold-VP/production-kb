import os
import logging
from pathlib import Path

from github import Github, GithubException

from config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH, KB_DIR

logger = logging.getLogger(__name__)


class GitHubStorage:
    def __init__(self):
        self._client = None
        self._repo   = None
        self._ready  = False
        self._init()

    def _init(self):
        if not GITHUB_TOKEN or not GITHUB_REPO:
            logger.warning("GitHub credentials not set — storage disabled.")
            return
        try:
            self._client = Github(GITHUB_TOKEN)
            self._repo   = self._client.get_repo(GITHUB_REPO)
            self._ready  = True
            logger.info("GitHub storage initialised: %s @ %s", GITHUB_REPO, GITHUB_BRANCH)
        except GithubException as e:
            logger.error("GitHub init failed: %s", e)

    # ------------------------------------------------------------------
    def pull_kb(self) -> int:
        """Download all .md files from GitHub repo into local KB_DIR.
        Returns number of files downloaded."""
        if not self._ready:
            logger.warning("pull_kb skipped — storage not ready.")
            return 0

        Path(KB_DIR).mkdir(parents=True, exist_ok=True)
        count = 0
        try:
            contents = self._repo.get_contents("kb", ref=GITHUB_BRANCH)
            # get_contents can return a single file or a list
            if not isinstance(contents, list):
                contents = [contents]

            for item in contents:
                if not item.name.endswith(".md"):
                    continue
                local_path = Path(KB_DIR) / item.name
                local_path.write_bytes(item.decoded_content)
                count += 1
                logger.info("Pulled: %s", item.name)

        except GithubException as e:
            # kb/ folder may not exist yet — that's fine
            logger.warning("pull_kb error (folder may be empty): %s", e)

        return count

    # ------------------------------------------------------------------
    def push_file(self, filepath: str, content: str) -> bool:
        """Commit a single file to GitHub. Creates or updates as needed."""
        if not self._ready:
            logger.warning("push_file skipped — storage not ready.")
            return False

        # GitHub path uses forward slashes relative to repo root
        rel_path = Path(filepath).as_posix()
        # Normalise: ensure it starts with kb/
        if not rel_path.startswith("kb/"):
            rel_path = "kb/" + Path(filepath).name

        try:
            try:
                existing = self._repo.get_contents(rel_path, ref=GITHUB_BRANCH)
                self._repo.update_file(
                    path    = rel_path,
                    message = f"update: {Path(filepath).name}",
                    content = content,
                    sha     = existing.sha,
                    branch  = GITHUB_BRANCH,
                )
                logger.info("Updated on GitHub: %s", rel_path)
            except GithubException:
                # File doesn't exist yet — create it
                self._repo.create_file(
                    path    = rel_path,
                    message = f"add: {Path(filepath).name}",
                    content = content,
                    branch  = GITHUB_BRANCH,
                )
                logger.info("Created on GitHub: %s", rel_path)
            return True

        except GithubException as e:
            logger.error("push_file failed for %s: %s", rel_path, e)
            return False

    # ------------------------------------------------------------------
    def list_files(self) -> list[dict]:
        """Return list of dicts {name, path, last_modified} for all .md files."""
        if not self._ready:
            return []
        try:
            contents = self._repo.get_contents("kb", ref=GITHUB_BRANCH)
            if not isinstance(contents, list):
                contents = [contents]

            result = []
            for item in contents:
                if not item.name.endswith(".md"):
                    continue
                # last_modified comes from the latest commit touching this file
                commits = self._repo.get_commits(path=item.path, sha=GITHUB_BRANCH)
                last_modified = None
                try:
                    last_modified = commits[0].commit.author.date
                except (IndexError, Exception):
                    pass
                result.append({
                    "name":          item.name,
                    "path":          item.path,
                    "last_modified": last_modified,
                })
            return result

        except GithubException as e:
            logger.error("list_files failed: %s", e)
            return []
