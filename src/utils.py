import os
import base64

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class ModelCrypto:
    def __init__(self, password: bytes = b"NuboMed", salt: bytes = None):
        """
        初始化加密器。

        :param password: 密码 (默认为 b"NuboMed")
        :param salt: 盐值 (默认为你提供的特定字节串)
        """
        self.password = password

        # 使用你指定的默认 Salt
        if salt is None:
            self.salt = b"\xc3*\xa2.\x8d\xbc3\xd0C.\xf5\x83\xa6\xba;o"
        else:
            self.salt = salt

        self.fernet = self._generate_key()

    def _generate_key(self) -> Fernet:
        """根据密码和盐生成 Fernet 密钥实例"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=1_200_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return Fernet(key)

    def encrypt_file(self, file_path: str, output_path: str = None) -> bool:
        """
        加密文件。

        :param file_path: 原文件路径
        :param output_path: 输出路径 (默认在原文件名后加 .enc)
        :return: 是否成功
        """
        if not os.path.exists(file_path):
            print(f"[错误] 文件不存在: {file_path}")
            return False

        try:
            if output_path is None:
                output_path = file_path + ".enc"

            with open(file_path, "rb") as fin:
                data = fin.read()

            encrypted_data = self.fernet.encrypt(data)

            with open(output_path, "wb") as fout:
                fout.write(encrypted_data)

            print(f"[加密成功] {file_path} -> {output_path}")
            return True
        except Exception as e:
            print(f"[加密错误] {e}")
            return False

    def decrypt_file(self, file_path: str, output_path: str = None) -> bool:
        """
        解密文件。

        :param file_path: 加密文件路径
        :param output_path: 输出路径 (默认去掉 .enc 后缀，如果没有 .enc 则加 .decrypted)
        :return: 是否成功
        """
        if not os.path.exists(file_path):
            print(f"[错误] 文件不存在: {file_path}")
            return False

        try:
            if output_path is None:
                if file_path.endswith(".enc"):
                    output_path = file_path[:-4]
                else:
                    output_path = file_path + ".decrypted"

            with open(file_path, "rb") as fin:
                ciphertext = fin.read()

            decrypted_data = self.fernet.decrypt(ciphertext)

            with open(output_path, "wb") as fout:
                fout.write(decrypted_data)

            print(f"[解密成功] {file_path} -> {output_path}")
            return True

        except InvalidToken:
            print(f"[解密失败] 密钥错误或文件已损坏: {file_path}")
            return False
        except Exception as e:
            print(f"[解密错误] {e}")
            return False


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 实例化 (使用默认的密码和 Salt)
    crypto_tool = ModelCrypto()

    # 2. 批量解密示例
    files_to_decrypt = ["best.bin.enc", "best.xml.enc"]

    print("--- 开始批量解密 ---")
    for f_path in files_to_decrypt:
        crypto_tool.decrypt_file(f_path)

    # 3. 加密示例 (如果你以后需要重新加密)
    # crypto_tool.encrypt_file("best.bin")
