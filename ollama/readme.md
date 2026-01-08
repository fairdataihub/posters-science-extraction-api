# Ollama

Create the network:

```bash
docker network create ollama-network
```

Start the ollama container:

```bash
docker compose -f ollama-compose.yml up -d
```

Use from any other Compose project by adding the following to the `networks` section:

```yaml
services:
  your-app:
    environment:
      - OLLAMA_HOST=http://ollama:11434
    networks:
      - ollama-network

networks:
  ollama-network:
    external: true
    name: ollama-network
```
