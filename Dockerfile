FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    openssh-server \
    libgl1 \
    && apt-get clean

COPY . .

EXPOSE 7860
EXPOSE 8888

RUN pip install --upgrade pip \
&& pip install --no-cache-dir -r requirements.txt

RUN jupyter server --generate-config \
    && echo "c.PasswordIdentityProvider.hashed_password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$PumRHL0BxA+w+sokTdEh9Q\$/AfZfhsfUKSGsqinhdkeIaM2bKnzr+znsXDJgUq2W2I'" \
    >> /root/.jupyter/jupyter_server_config.py

RUN chmod +x /app/start.sh

CMD ["/app/start.sh", "jupyter"]