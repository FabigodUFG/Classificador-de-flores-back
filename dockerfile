# Usar uma imagem base do Python
FROM python:3.9-slim

# Configurar o diretório de trabalho
WORKDIR /app

# Copiar os arquivos do backend para o contêiner
COPY . .

#permissão para escrever
RUN chmod -R 777 /usr/local/lib/python3.9/site-packages

# Instalar dependências do backend
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta do backend
EXPOSE 5000

# Comando para iniciar o backend
CMD ["python", "main.py"]