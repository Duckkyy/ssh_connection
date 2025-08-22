import paramiko
import os
from getpass import getpass

# ...existing code for connect_ssh and transfer_files functions...
def connect_ssh(hostname, username, password=None, key_filename=None):
    """Establish SSH connection to remote server"""
    try:
        # Initialize SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to remote server
        if key_filename:
            ssh.connect(hostname, username=username, key_filename=key_filename)
        else:
            ssh.connect(hostname, username=username, password=password)
        
        return ssh
    except Exception as e:
        print(f"Failed to connect to SSH: {str(e)}")
        return None

def transfer_files(sftp, local_path, remote_path, direction='upload'):
    """Transfer files between local and remote server"""
    try:
        if direction == 'upload':
            sftp.put(local_path, remote_path)
            print(f"Successfully uploaded {local_path} to {remote_path}")
        else:
            sftp.get(remote_path, local_path)
            print(f"Successfully downloaded {remote_path} to {local_path}")
    except Exception as e:
        print(f"File transfer failed: {str(e)}")

def execute_command(ssh, command):
    """Execute a command on the remote server"""
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if error:
            print("Error:", error)
        if output:
            print("Output:", output)
            
        return output, error
    except Exception as e:
        print(f"Command execution failed: {str(e)}")
        return None, str(e)
    

def main():
    # SSH connection details
    hostname = input("Enter server address: ")
    username = input("Enter username: ")
    use_key = input("Use SSH key? (yes/no): ").lower() == 'yes'
    
    if use_key:
        key_path = input("Enter path to SSH key: ")
        ssh = connect_ssh(hostname, username, key_filename=key_path)
    else:
        password = getpass("Enter password: ")
        ssh = connect_ssh(hostname, username, password=password)
    
    if ssh:
        try:
            while True:
                print("\n1. Upload file")
                print("2. Download file")
                print("3. Execute command")
                print("4. Run Python script")
                print("5. Exit")
                choice = input("Select option (1-5): ")
                
                if choice == '5':
                    break
                    
                if choice in ['1', '2']:
                    sftp = ssh.open_sftp()
                    local_path = input("Enter local file path: ")
                    remote_path = input("Enter remote file path: ")
                    
                    if choice == '1':
                        transfer_files(sftp, local_path, remote_path, 'upload')
                    else:
                        transfer_files(sftp, local_path, remote_path, 'download')
                    sftp.close()
                
                elif choice == '3':
                    command = input("Enter command to execute: ")
                    execute_command(ssh, command)
                
                elif choice == '4':
                    script_path = input("Enter remote Python script path: ")
                    python_cmd = f"python3 {script_path}"
                    execute_command(ssh, python_cmd)
                
                elif choice == '6':
                    command = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate wan_gpu && cd ~/member_files/dai/Index-anisora/anisoraV2_gpu && "
                    python_cmd = "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 43210 generate-pi-i2v.py --task i2v-14B --size 960*544 --ckpt_dir Wan2.1-I2V-14B-480P --image image_video --prompt inference.txt --dit_fsdp --t5_fsdp --ulysses_size 4 --base_seed 4096 --frame_num 49"
                    execute_command(ssh, command + python_cmd)
            
            # Close connection
            ssh.close()
            print("SSH connection closed")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
if __name__ == "__main__":
    main()