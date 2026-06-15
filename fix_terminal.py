import os
import winreg

def add_powershell_to_path():
    try:
        # Open the registry key for the current user's environment variables
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment', 0, winreg.KEY_READ | winreg.KEY_WRITE)
        try:
            current_path, _ = winreg.QueryValueEx(key, 'Path')
        except FileNotFoundError:
            current_path = ""
            
        ps_path = r"C:\Windows\System32\WindowsPowerShell\v1.0"
        
        if ps_path.lower() not in current_path.lower():
            new_path = current_path + (";" if current_path and not current_path.endswith(";") else "") + ps_path
            winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
            print("SUCCESS! PowerShell path successfully added to your user environment variables!")
            print("IMPORTANT: You must fully close and restart PyCharm for the terminal to recognize the change.")
        else:
            print("PowerShell is already in your user PATH.")
            
        winreg.CloseKey(key)
    except Exception as e:
        print(f"Failed to update PATH: {e}")

if __name__ == "__main__":
    print("Fixing Terminal PATH...")
    add_powershell_to_path()
