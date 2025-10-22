SHENGLONG'S QUANTUM GROUP LIVE GITHUB

TO "INSTALL" THE REPO
 * Install git for terminal: 
 * * Windows (use "winget install git.git" then restart terminal)
 * * macOS (use "brew install git")
 * * linux (you should have this already if you use linux)
 * Start a terminal in the directory you want to clone the repo to
 * Run "git clone https://github.com/nathanielasun/"
 * Then run "git init"
 * Run a github pull to make sure all files are updated "git pull origin main"
 * * This should say "Already up to date."

You should now see the files in your directory

VERY IMPORTANT!!!!
 - Best practice is to push ONLY THE EDITS ON YOUR FILE
 - Use "git add yournotebook.ipynb" instead of "git add ."
 - This can lead to merge hell if you don't, as other's code will be overwritten

To push
 - git add yourfile.ipynb ("git add ." works, but updates all files concurrently)
 - git commit -m "message"
 - git pull --rebase (this ensures your local edits are saved while pulling everyone else's)
 - git push -u origin main

LET ME KNOW BEFORE YOU MAKE ANY MAJOR EDITS TO COLLAB NOTEBOOKS (work in your personal one)

THANKS
Nate

Email or text me if you have questions
(nathanielsun@tamu.edu ; 832-352-5772)