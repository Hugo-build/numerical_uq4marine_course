# Linux and macOS shell basics, Git, and GitHub CLI

A beginner's guide to navigating folders, managing files, and keeping project history.
The examples use commands shared by Bash and Zsh on Linux and macOS, except
where a shell-specific command is explicitly marked.

## 1. Getting started

Open **Terminal** on macOS or your Linux terminal application. A terminal is the
window you type into; the shell is the program that interprets your commands.

Type a command and press **Enter**. Run examples one line at a time while learning.
Code blocks below omit the prompt symbol (`$`), so you can copy just the commands.
Lines beginning with `#` are explanatory comments.

A command usually has this structure:

```sh
command options arguments
```

For example, `ls -l Documents` runs `ls`, uses the `-l` option for a detailed listing,
and lists the folder named `Documents`.

Replace example names such as `YOUR-USERNAME` with your own values. Command names
are case-sensitive. Use the exact spelling of file names; Linux file systems are
usually case-sensitive, while macOS configurations vary.

### Useful keyboard shortcuts

| Key | Action |
|---|---|
| Tab | Complete a command or path; behavior depends on shell configuration |
| Up / Down | Browse previous commands |
| Ctrl+C | Interrupt the running command |
| Ctrl+A / Ctrl+E | Move to the beginning/end of the line in common default key bindings |
| Ctrl+L | Clear the terminal display |
| `q` | Exit viewers such as `less` and `man` |

## 2. Paths and navigation: `pwd`, `cd`, and `ls`

A **directory** is a folder. Your **working directory** is the folder in which
commands currently run.

| Command | Meaning |
|---|---|
| `pwd` | Print the current directory's path |
| `ls` | List visible files and directories |
| `ls -l` | List details, including permissions, owner, size, and modification time |
| `ls -a` | Include hidden names, which begin with `.` |
| `ls -lh` | Show details with readable sizes, such as KB or MB |
| `ls -la` | Show details including hidden entries |
| `cd Documents` | Enter the `Documents` folder inside the current directory |
| `cd ..` | Go to the parent directory |
| `cd ~` or `cd` | Go to your home directory |
| `cd -` | Return to the previous working directory |

Common path notation:

| Symbol | Meaning | Example |
|---|---|---|
| `/` | File-system root; also separates path components | `/tmp` |
| `~` | Your home directory | `~/Documents` |
| `.` | Current directory | `./notes.txt` |
| `..` | Parent directory | `../data` |

An **absolute path** starts at `/`, such as `/Users/alex/Documents` on macOS or
`/home/alex/Documents` on Linux. A **relative path** starts from your working
directory, such as `data/input.txt`.

```sh
pwd
cd ~
ls -la
cd "Documents/My Project"
cd ..
```

The quoted example requires that folder to exist. Quote paths containing spaces.
For a home-relative path with spaces, use `cd "$HOME/Documents/My Project"`;
`"~/Documents/My Project"` does not expand `~`.

## 3. Create, copy, move, and delete files

| Command | Meaning |
|---|---|
| `mkdir project` | Create a directory |
| `mkdir -p project/data/raw` | Create missing parent directories too |
| `touch notes.txt` | Create an empty file if absent; otherwise update its timestamps |
| `cp notes.txt notes-copy.txt` | Copy a file |
| `cp -R data data-copy` | Copy a directory and its contents |
| `mv notes-copy.txt backup.txt` | Rename a file |
| `mv backup.txt data/` | Move a file into an existing directory |
| `rm -i backup.txt` | Ask before deleting a file |
| `rmdir empty-folder` | Delete an empty directory |
| `rm -r old-folder` | Delete a directory and its contents recursively |

`cp` and `mv` can overwrite existing destinations. Use `cp -i` or `mv -i` to ask
before overwriting. When a destination directory exists, a source is normally
placed inside it: `cp -R data archive` creates `archive/data` if `archive` exists.

**Deletion with `rm` normally bypasses Trash and has no built-in undo.** Check
`pwd` and `ls` before deleting. Avoid `rm -rf` while learning: `-r` deletes
recursively, and `-f` suppresses normal prompts and some errors.

## 4. Read and edit text

| Command | Meaning |
|---|---|
| `cat notes.txt` | Print the entire file |
| `less notes.txt` | Read a file one screen at a time; press `q` to leave |
| `head -n 10 notes.txt` | Show the first ten lines |
| `tail -n 10 notes.txt` | Show the last ten lines |
| `tail -f app.log` | Watch appended log lines; stop with Ctrl+C |
| `wc -l notes.txt` | Count newline characters, normally equivalent to lines in text files |
| `nano notes.txt` | Edit a file in Nano, if installed |

In Nano, **Ctrl+O**, then **Enter**, saves the file; **Ctrl+X** exits.

### Redirect output and connect commands

```sh
printf 'First note\n' > notes.txt
printf 'Second note\n' >> notes.txt
cat notes.txt
ls -la > directory-list.txt
cat notes.txt | wc -l
```

- `>` writes output to a file, **replacing its previous contents**.
- `>>` appends output to a file.
- `|` sends one command's standard output into the next command's standard input.
- These examples redirect normal output; error messages still appear in the terminal.

`printf` is useful when you want explicit line breaks (`\n`).

## 5. Find files and search their contents

```sh
# Find Markdown files below the current directory.
find . -type f -name '*.md'

# Find a word in one file; -n shows line numbers.
grep -n 'error' app.log

# Search recursively, ignoring letter case.
grep -Rin 'todo' .

# List text files in the current directory.
ls *.txt
```

`find` searches names and file properties. `grep` searches file contents.
Unquoted `*.txt` is expanded by the shell to matching paths. Quote `'*.md'` in the
`find` example so `find` receives the pattern itself. If a wildcard matches no
files, Bash and Zsh can behave differently; default Zsh reports `no matches found`.

## 6. Help, programs, processes, and permissions

| Command | Meaning |
|---|---|
| `man ls` | Open the manual for `ls`; press `q` to exit |
| `command -v git` | Find how the shell would resolve `git` |
| `which git` | Locate `git`; exact behavior depends on the shell and installed `which` |
| `which -a python3` | Show all matching locations; supported by Zsh and many external `which` utilities |
| `where python3` | Zsh only: show all matching command definitions and executable locations |
| `type -a python3` | Bash and Zsh: describe all matching definitions and executable locations |
| `history` | Display command history |
| `clear` | Clear the terminal display |
| `df -h` | Show file-system disk usage |
| `du -sh .` | Summarize the current directory's size |
| `ps` | Show a basic process listing |
| `top` | Show an updating process view; press `q` to exit |
| `kill PID` | Request termination of a process; replace `PID` with its numeric ID |
| `chmod u+x script.sh` | Give the file's owner permission to execute a script |
| `./script.sh` | Execute a script in the current directory, if configured to run |

### Locate commands: `which`, `where`, and `PATH`

Use these commands to check which installation of a program your shell finds,
especially when you have multiple Python or Git installations:

```sh
# Bash or Zsh: check how the current shell resolves a command.
command -v python3
type -a python3

# Locate a program and check for multiple installations.
which python3
which -a python3

# Zsh only: list every matching definition or executable location.
where python3
where git

# Display the directories searched for external commands.
echo "$PATH"
```

`PATH` is a colon-separated list of directories. For external programs, the
shell normally searches these directories from left to right. For example,
`/usr/local/bin:/usr/bin:/bin` puts `/usr/local/bin` first. Output such as
`/usr/bin/python3` is the executable's location; your actual path may differ.

In Zsh, `which` and `where` are builtins that also recognize aliases and
functions. In Bash, `which` is usually an external utility and may miss those
definitions. Prefer `command -v` to check the current shell's resolution and
`type -a` to inspect all matches. Bash has no built-in `where`; use `type -a`
there. See the [Zsh builtin reference](https://zsh.sourceforge.io/Doc/Release/Shell-Builtin-Commands.html)
and [Bash builtin reference](https://www.gnu.org/software/bash/manual/html_node/Bash-Builtins.html).

These commands locate commands, rather than arbitrary project files. Use
`find` from section 5 to search for files. If `command -v python3` produces no
output, check that Python is installed and its directory is in `PATH`.

In a permission string such as `-rw-r--r--`, the first character indicates the
file type. The next three groups describe owner, group, and others: `r` means
read, `w` means write, and `x` means execute (or directory traversal).

`sudo command` runs a command as another user, usually the administrator. Use it
only when you understand why administrative access is required. Password entry
usually displays no characters; that is normal. Ordinary project work should
not require `sudo`.

### Variables and successful-command chaining

```sh
echo "$HOME"
export COURSE_NAME="numerical engineering"
echo "$COURSE_NAME"
mkdir -p practice && cd practice
```

`export` makes a variable available to programs launched from this shell. It lasts
for this session unless you configure it in a shell startup file. `&&` runs the
second command only if the first succeeds.

## 7. Linux and macOS differences

The basic commands above work on both systems, but some options differ between
GNU utilities commonly used on Linux and BSD utilities bundled with macOS.
Use `man command` to check the options installed on your computer. For example,
`--help` is common on Linux but is not supported by every macOS command.

| Task | macOS | Linux |
|---|---|---|
| Typical home path | `/Users/username` | `/home/username` |
| Open the current folder in a graphical app | `open .` | `xdg-open .` on desktops where installed |
| Install software | Homebrew, if installed | Your distribution's package manager |

Linux has multiple distributions; commands for Ubuntu/Debian do not automatically
apply to Fedora, Arch, or other distributions.

## 8. Practice: a small shell project

Start in a fresh practice directory. Run this exercise once; the first `mkdir`
will report an error if that name already exists, so choose a new name in that case.

```sh
cd ~
mkdir shell-practice
cd shell-practice
mkdir -p data/raw results
printf 'Learning shell commands\n' > notes.txt
printf 'Next: learn Git\n' >> notes.txt
cp notes.txt notes-backup.txt
mv notes-backup.txt results/
ls -la
cat notes.txt
grep -n 'Git' notes.txt
find . -type f
pwd
```

You should have `notes.txt`, `results/notes-backup.txt`, and an empty `data/raw`
directory. The search should find `Git` on line 2 of `notes.txt`.

## 9. Git, GitHub, and the GitHub client

- **Git** tracks file changes and saves project history locally.
- **GitHub** hosts Git repositories online and provides collaboration tools.
- **GitHub CLI**, the `gh` client used here, manages GitHub repositories, pull
  requests, and issues from your terminal. See the [official CLI manual](https://cli.github.com/manual/gh).
- **GitHub Desktop** is a separate graphical client. The examples here use the
  command-line tools `git` and `gh`.

A **repository** contains a project and its Git history. A **commit** is a saved
snapshot of staged changes. A **branch** is a named line of development. A
**remote** is a configured connection to another repository, often on GitHub.
A **pull request (PR)** asks others to review and merge changes from a branch.

### Installation and identity

First check whether the tools are already installed:

```sh
git --version
gh --version
```

On macOS with Homebrew already installed:

```sh
brew install git gh
```

On Ubuntu/Debian, Git can be installed with:

```sh
sudo apt update
sudo apt install git
```

For GitHub CLI on Linux, follow the installation instructions for your
distribution in the [official GitHub CLI repository](https://github.com/cli/cli#installation).
For Git installation alternatives, see [Git downloads](https://git-scm.com/downloads).

Set the identity recorded in your commits, replacing these sample values:

```sh
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

`--global` sets a default for all your repositories. Omit it when setting identity
for just the current repository. These values identify commit authors; they do
not log you into GitHub. Use a GitHub-provided no-reply address if you want to
keep your personal email out of published commits.

## 10. Save your first project with Git

Continue in `~/shell-practice` from the shell exercise. For a different project,
first `cd` into that project's directory.

```sh
git init -b main
git status
git add notes.txt
git diff --staged
git commit -m "Add initial shell notes"
git log --oneline
```

`git init -b main` creates a repository with an initial branch named `main`.
Run it once when starting a new repository. Cloning an existing repository
already creates the local repository for you.

The normal saving sequence is:

```text
edit files → git add → git commit → git push
```

The **working tree** contains your current files. `git add` places the current
version of selected changes in the **staging area**. `git commit` saves the
staged changes into local history. If you edit a file after staging it, stage it
again to include the newer edit. `git push` later uploads commits to a remote.
These concepts and commands are covered in the [Git reference](https://git-scm.com/docs).

### Everyday Git commands

| Command | Purpose |
|---|---|
| `git status` | Show the current branch, modified files, and staged changes |
| `git diff` | Show unstaged changes in tracked files |
| `git diff --staged` | Show changes prepared for the next commit |
| `git add notes.txt` | Stage a specific file |
| `git add .` | Stage changes below the current directory, including deletions |
| `git commit -m "Describe the change"` | Save staged changes locally |
| `git log --oneline -n 5` | Show the five most recent commits |
| `git branch` | List local branches |
| `git switch -c update-notes` | Create and switch to a new branch |
| `git switch main` | Switch to an existing branch |
| `git remote -v` | Show configured remote names and addresses |
| `git fetch origin` | Download remote history without merging it into your current branch |
| `git pull --ff-only` | Fetch and update the current branch only if it can fast-forward |
| `git push` | Upload commits using the configured upstream |

`git diff` does not display the contents of untracked files; `git status` lists
them. Review `git status` before using `git add .` so you do not stage unrelated
files or private data.

`git pull --ff-only` refuses to combine diverged local and remote histories. If
it fails, inspect the changes and agree on a merge or rebase approach with your
collaborators. See the [Git pull documentation](https://git-scm.com/docs/git-pull).

### Ignore generated files

Create a file named `.gitignore` in the repository root. For a Python project,
it might contain:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
.env
```

Commit `.gitignore` so collaborators use the same rules. Ignore rules do not
stop tracking files that are already committed. Do not commit passwords, access
tokens, or private keys; ignoring a file does not remove it from past history.

### Unstage or discard changes

After your repository has at least one commit:

```sh
# Unstage this file while keeping your edits.
git restore --staged notes.txt

# Discard unstaged edits to this tracked file.
git restore notes.txt
```

**The second command can permanently lose uncommitted work.** With its default
settings, it restores the file from the staging area. See the
[Git restore documentation](https://git-scm.com/docs/git-restore).

## 11. Use the GitHub CLI (`gh`)

### Log in

```sh
gh auth login
gh auth status
```

Follow the prompts to choose GitHub.com, a Git protocol, and browser login.
For a beginner, HTTPS is a straightforward choice; accept the Git authentication
setup when offered. If you need to configure Git to use your CLI login later,
run `gh auth setup-git`. See [GitHub CLI authentication](https://cli.github.com/manual/gh_auth_login).

### Option A: publish the practice repository

From `~/shell-practice`, after making the initial commit:

```sh
gh repo create shell-practice --private --source=. --remote=origin --push
gh browse
```

This creates a **private repository on GitHub**, adds a remote named `origin`,
and uploads your local commits. It requires an unused repository name and no
existing remote named `origin`. Change the name if needed. Use `--public`
instead of `--private` only when you intend to publish the project publicly.
See [repository creation options](https://cli.github.com/manual/gh_repo_create).

### Option B: clone an existing repository

Use this instead of initializing a new repository when the project already
exists on GitHub. Replace `YOUR-USERNAME/REPOSITORY` with the actual owner/name:

```sh
cd ~
gh repo clone YOUR-USERNAME/REPOSITORY
cd REPOSITORY
git status
```

The Git equivalent is `git clone https://github.com/YOUR-USERNAME/REPOSITORY.git`.
Cloning downloads the files and history and usually configures `origin`.
Private repositories require access. See the [CLI examples](https://cli.github.com/manual/examples).

### Make a change and open a pull request

For the published practice repository, start on `main` with a clean working
tree (`git status` should show no pending changes):

```sh
git switch main
git pull --ff-only
git switch -c improve-notes
printf 'Practice a little every day.\n' >> notes.txt
git diff
git add notes.txt
git diff --staged
git commit -m "Add a practice reminder"
git push -u origin improve-notes
gh pr create --base main --title "Improve shell notes" --body "Add a daily practice reminder."
gh pr view --web
```

`-u` sets the branch's upstream, so later pushes on that branch can use `git push`.
The PR command creates a review request on GitHub; follow your project's review
process before merging it. See [pull request creation](https://cli.github.com/manual/gh_pr_create).

After the PR has been merged on GitHub:

```sh
git switch main
git pull --ff-only
```

Other projects may use a default branch name such as `master`; substitute their
actual branch name where this guide uses `main`.

### More useful GitHub commands

Run these inside a repository linked to GitHub. Commands containing `12` use an
example issue or PR number; replace it with a real one.

| Command | Purpose |
|---|---|
| `gh browse` | Open the repository in your browser |
| `gh repo view` | Show repository information |
| `gh pr list` | List open pull requests |
| `gh pr view 12` | Show a pull request |
| `gh pr checkout 12` | Check out a pull request locally; save current edits first |
| `gh issue list` | List open issues |
| `gh issue view 12` | Show an issue |
| `gh issue create` | Follow prompts to create an issue on GitHub |
| `gh help` | Show CLI help |
| `gh pr create --help` | Show help for a specific command |

Command details are available in the [GitHub CLI manual](https://cli.github.com/manual/gh).

## 12. Common problems

| Message or situation | What to check |
|---|---|
| `command not found` | Check spelling and installation; use `command -v NAME`, `type -a NAME`, and `echo "$PATH"` to inspect command lookup |
| `No such file or directory` | Use `pwd` and `ls`; check the path and quote spaces |
| `Permission denied` | Check ownership and permissions; determine the cause before using `sudo` |
| `not a git repository` | `cd` into the cloned/initialized project, or initialize a genuinely new project |
| `nothing to commit` | Use `git status`; confirm files were changed and staged and are not ignored |
| GitHub authentication error | Run `gh auth status`; log in again if necessary and confirm repository access |
| Push rejected | Fetch and inspect remote changes; update your branch using the project's workflow |
| Merge conflict | Edit the conflicted files, remove conflict markers, stage the resolved files, and follow Git's instructions to continue |

## 13. Quick reference

```sh
# Navigate and inspect.
pwd
ls -la
cd ..

# Locate installed commands (Bash or Zsh).
command -v git
which python3
type -a python3
# Zsh only: where python3

# Create and read.
mkdir new-folder
touch new-folder/notes.txt
cat new-folder/notes.txt

# Save project changes locally (inside a repository).
git status
git diff
git add notes.txt
git diff --staged
git commit -m "Update notes"

# Collaborate (with a configured remote and upstream).
git pull --ff-only
git push
gh pr list
```

The final block is a reminder, not a single script: review your current branch
and working tree before pulling or pushing.
