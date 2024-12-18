
import paramiko
from scp import SCPClient

def upload_file():
    """
    上传文件
    :return:
    """
    ssh_client = paramiko.SSHClient()
    pkey='/home/crypto/.ssh/id_rsa'
    key=paramiko.RSAKey.from_private_key_file(pkey)

    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect('ec2-3-99-203-76.ca-central-1.compute.amazonaws.com',username = 'ubuntu',pkey=key)
    scp_client = SCPClient(ssh_client.get_transport(), socket_timeout=15.0)
    try:
        scp_client.put("a", "b")
    except FileNotFoundError as e:
        print(e)
        print("系统找不到指定文件" + "a")
    else:
        print("文件上传成功")
    ssh_client.close()
    
if __name__ == "__main__":
    upload_file()
