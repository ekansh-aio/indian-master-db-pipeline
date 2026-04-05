# Windows Setup Guide

## 🪟 Running on Windows

You have **three options** to run the pipeline on Windows:

### Option 1: PowerShell (Recommended) ⭐

**PowerShell has colored output and better error handling**

```powershell
# Run directly
.\run.ps1

# Or if you get execution policy error:
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

**Note:** If you get "execution policy" errors, run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Option 2: Batch File

**Simple and works everywhere**

```cmd
run.bat
```

Double-click `run.bat` or run from Command Prompt.

### Option 3: Direct Python

**Manual but most reliable**

```cmd
python pipeline.py
```

## 🚀 Complete Setup Steps (Windows)

### Step 1: Install Python

1. Download Python 3.8+ from https://www.python.org/downloads/
2. **Important:** Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   ```

### Step 2: Install Dependencies

Open Command Prompt or PowerShell in the project directory:

```cmd
pip install -r requirements.txt
```

**If you get SSL errors:**
```cmd
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Step 3: Configure Environment

```cmd
copy .env.example .env
notepad .env
```

Edit the `.env` file with your Azure credentials:

```env
# Azure Data Lake Storage
ADLS_ACCOUNT_NAME=your_storage_account
ADLS_ACCOUNT_KEY=your_key_here
ADLS_CONTAINER_NAME=raw
ADLS_INPUT_PATH=raw/newapp/dataset=bill/bill/queensland/1992

# Azure AI Search
SEARCH_ENDPOINT=https://your-search.search.windows.net
SEARCH_KEY=your_admin_key_here
INDEX_NAME=legal-documents-index
```

### Step 4: Run the Pipeline

**Using PowerShell (Recommended):**
```powershell
.\run.ps1
```

**Using Batch:**
```cmd
run.bat
```

**Using Python directly:**
```cmd
python pipeline.py
```

## 🔧 Troubleshooting Windows Issues

### Issue: "python is not recognized"

**Solution:**
1. Reinstall Python with "Add to PATH" checked
2. OR add Python to PATH manually:
   - Search for "Environment Variables" in Windows
   - Add Python install directory to PATH (e.g., `C:\Python39\`)
   - Add Scripts directory too (e.g., `C:\Python39\Scripts\`)

### Issue: "pip is not recognized"

**Solution:**
```cmd
python -m pip install -r requirements.txt
```

### Issue: PowerShell execution policy error

**Solution:**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run with bypass:
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### Issue: SSL Certificate errors during pip install

**Solution:**
```cmd
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: "Access denied" errors

**Solution:**
- Run Command Prompt or PowerShell as Administrator
- OR install packages for current user only:
  ```cmd
  pip install --user -r requirements.txt
  ```

### Issue: Long path errors

**Solution:**
Enable long paths in Windows:
1. Run `gpedit.msc`
2. Navigate to: Computer Configuration > Administrative Templates > System > Filesystem
3. Enable "Enable Win32 long paths"

### Issue: Can't edit .env file

**Solution:**
```cmd
# Use notepad
notepad .env

# Or use VSCode
code .env

# Or use any text editor
```

## 📊 File Paths on Windows

The pipeline creates these directories (with Windows backslashes):

```
output\
├── intermediate\
│   ├── all_chunks.json
│   └── top_k_chunks.json
└── final\
    └── processing_stats.json

pipeline.log
```

## 🎯 Running in Different Environments

### Command Prompt
```cmd
cd C:\path\to\project
python pipeline.py
```

### PowerShell
```powershell
cd C:\path\to\project
python pipeline.py
```

### PowerShell ISE
```powershell
Set-Location "C:\path\to\project"
python pipeline.py
```

### VSCode Terminal
```bash
# Works with both PowerShell and CMD
python pipeline.py
```

### Windows Terminal
```powershell
# Supports both PowerShell and CMD
python pipeline.py
```

## 🔐 Windows-Specific .env Notes

**Use forward slashes OR escaped backslashes in paths:**

✅ Good:
```env
ADLS_INPUT_PATH=raw/newapp/dataset=bill
```

✅ Also Good:
```env
ADLS_INPUT_PATH=raw\\newapp\\dataset=bill
```

❌ Bad:
```env
ADLS_INPUT_PATH=raw\newapp\dataset=bill
```

## 🎨 Better Terminal Experience

**For colored output and better experience:**

1. **Install Windows Terminal** (free from Microsoft Store)
2. **Use PowerShell 7+** instead of Windows PowerShell 5.1
3. **Run the PowerShell script** (run.ps1) for colored output

## 📝 Quick Reference

### Start Pipeline
| Method | Command |
|--------|---------|
| PowerShell | `.\run.ps1` |
| Batch | `run.bat` |
| Direct | `python pipeline.py` |

### Check Logs
```cmd
type pipeline.log
# or
notepad pipeline.log
# or
code pipeline.log
```

### View Output Files
```cmd
# Open in default JSON viewer
start output\intermediate\all_chunks.json

# Or use notepad
notepad output\intermediate\all_chunks.json

# Or use VSCode
code output\intermediate\all_chunks.json
```

### Test Configuration
```cmd
python -c "from config import validate_config; validate_config(); print('Config OK')"
```

### Check Dependencies
```cmd
pip list | findstr "azure sentence"
```

## 🚀 Performance Tips for Windows

1. **Use SSD** for output directory
2. **Add exclusions** in Windows Defender for project directory
3. **Close unnecessary applications** during processing
4. **Increase batch size** in config for better performance:
   ```env
   BATCH_SIZE=20
   NUM_WORKERS=8
   ```

## 🔄 Environment Variables (Windows Alternative)

If you don't want to use .env file, you can set environment variables in Windows:

**Command Prompt:**
```cmd
set ADLS_ACCOUNT_NAME=your_account
set ADLS_ACCOUNT_KEY=your_key
python pipeline.py
```

**PowerShell:**
```powershell
$env:ADLS_ACCOUNT_NAME="your_account"
$env:ADLS_ACCOUNT_KEY="your_key"
python pipeline.py
```

**System Environment Variables:**
1. Search "Environment Variables" in Windows
2. Click "Environment Variables"
3. Add each variable under "User variables"
4. Restart terminal/IDE

## 📦 Installing on Corporate Windows (Proxy/Firewall)

If you're behind a corporate proxy:

```cmd
# Set proxy
set HTTP_PROXY=http://proxy.company.com:8080
set HTTPS_PROXY=http://proxy.company.com:8080

# Install with proxy
pip install -r requirements.txt

# Or specify proxy directly
pip install --proxy http://proxy.company.com:8080 -r requirements.txt
```

## 🎓 VSCode Integration (Recommended)

**Running from VSCode:**

1. Open project folder in VSCode
2. Open integrated terminal (Ctrl+`)
3. Select PowerShell or Command Prompt
4. Run: `python pipeline.py`

**Debug in VSCode:**

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Pipeline",
            "type": "python",
            "request": "launch",
            "program": "pipeline.py",
            "console": "integratedTerminal"
        }
    ]
}
```

Press F5 to debug!

## ✅ Verification Checklist

Before running, verify:

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Dependencies installed (`pip list`)
- [ ] .env file created and configured
- [ ] Azure credentials are correct
- [ ] Terminal is in project directory (`cd C:\path\to\project`)

## 🆘 Still Having Issues?

1. Check `pipeline.log` for detailed errors
2. Run with debug logging:
   ```cmd
   set LOG_LEVEL=DEBUG
   python pipeline.py
   ```
3. Verify Azure connection:
   ```cmd
   python -c "from adls_fetcher import ADLSFetcher; print('ADLS module OK')"
   ```

## 💡 Pro Tips

1. **Use Windows Terminal** for better experience
2. **Pin project folder** to Quick Access
3. **Create desktop shortcut** to run.bat for quick access
4. **Use VSCode** for editing .env and viewing logs
5. **Enable long paths** to avoid path length issues
6. **Add project to Windows Defender exclusions** for faster processing

---

**Ready to go? Run this:**

```powershell
.\run.ps1
```

or

```cmd
run.bat
```

🚀 Good luck!
