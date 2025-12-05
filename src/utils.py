import base64
from pathlib import Path
from typing import Union

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class ModelCrypto:
    def __init__(self, password: bytes = b"NuboMed", salt: bytes = None):
        """
        初始化加密器。

        :param password: 密码 (默认为 b"NuboMed")
        :param salt: 盐值
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

    def encrypt_file(self, file_path: Union[str, Path]) -> bool:
        """
        加密文件。

        :param file_path: 原文件路径 (str 或 Path 对象)
        :return: 是否成功
        """
        input_path = Path(file_path)

        if not input_path.exists():
            print(f"[错误] 文件不存在: {input_path}")
            return False

        try:
            # 在原文件名（包含后缀）后面直接追加 .enc，例如 best.bin -> best.bin.enc
            output_path = input_path.with_name(input_path.name + ".enc")
            data = input_path.read_bytes()
            encrypted_data = self.fernet.encrypt(data)
            output_path.write_bytes(encrypted_data)
            print(f"[加密成功] {input_path} -> {output_path}")
            return True
        except Exception as e:
            print(f"[加密错误] {e}")
            return False

    def decrypt_file(self, file_path: Union[str, Path]) -> bool:
        """
        解密文件。

        :param file_path: 加密文件路径
        :return: 是否成功
        """
        input_path = Path(file_path)

        if not input_path.exists():
            print(f"[错误] 文件不存在: {input_path}")
            return False

        try:
            # 去掉最后一个后缀 (.enc)
            output_path = input_path.with_suffix("")

            ciphertext = input_path.read_bytes()
            decrypted_data = self.fernet.decrypt(ciphertext)
            output_path.write_bytes(decrypted_data)

            print(f"[解密成功] {input_path} -> {output_path}")
            return True

        except InvalidToken:
            print(f"[解密失败] 密钥错误或文件已损坏: {input_path}")
            return False
        except Exception as e:
            print(f"[解密错误] {e}")
            return False

    def decrypt_to_bytes(self, file_path: Union[str, Path]) -> bytes:
        """
        解密文件并直接返回二进制数据，不写入硬盘。

        :param file_path: 加密文件路径
        :return: 解密后的 bytes 数据
        """
        input_path = Path(file_path)

        if not input_path.exists():
            raise FileNotFoundError(f"[错误] 文件不存在: {input_path}")

        try:
            ciphertext = input_path.read_bytes()
            decrypted_data = self.fernet.decrypt(ciphertext)
            return decrypted_data

        except InvalidToken:
            raise ValueError(f"[解密失败] 密钥错误或文件已损坏: {input_path}")
        except Exception as e:
            raise RuntimeError(f"[解密错误] {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    crypto_tool = ModelCrypto()

    base_dir = Path("../model")

    target_files = [base_dir / "best.bin", base_dir / "best.xml"]

    for f_path in target_files:
        if f_path.exists():
            crypto_tool.encrypt_file(f_path)
        else:
            print(f"跳过（未找到）: {f_path}")
