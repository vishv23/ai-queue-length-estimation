@echo off
set "PATH=C:\Users\Vishv\AppData\Local\Programs\MinGit\cmd;%PATH%"
echo =======================================================
echo    UPLOAD PROJECT TO GITHUB (AI Queue Length Estimation)
echo =======================================================
echo.
echo 1. Go to https://github.com/new in your browser.
echo 2. Create a new repository named: ai-queue-length-estimation
echo 3. Keep it Public and do NOT check "Add a README file".
echo 4. Copy the repository URL.
echo.
set /p REPO_URL="Paste your GitHub Repository URL here: "

if "%REPO_URL%"=="" (
    echo [Error] No repository URL was entered.
    pause
    exit /b
)

git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo.
echo Uploading project to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Project successfully uploaded to GitHub!
    echo Share this link with your teacher:
    echo %REPO_URL%
    echo =======================================================
) else (
    echo.
    echo [Notice] If prompted by GitHub, sign in via browser or Personal Access Token.
)
echo.
pause
