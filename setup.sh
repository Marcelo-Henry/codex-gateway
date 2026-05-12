#!/bin/bash
cd "$(dirname "$0")"

# Colors
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RESET='\033[0m'

# Load existing .env values (if file exists)
if [ -f .env ]; then
    source .env 2>/dev/null
fi

needs_setup=false

# Check which required vars are missing or empty
if [ -z "$PROXY_API_KEY" ] || [ "$PROXY_API_KEY" = "my-super-secret-password-123" ]; then
    needs_proxy_key=true
    needs_setup=true
else
    needs_proxy_key=false
fi

if [ -z "$SERVER_HOST" ]; then
    needs_host=true
    needs_setup=true
else
    needs_host=false
fi

if [ -z "$KIRO_CLI_DB_FILE" ]; then
    needs_sqlite=true
    needs_setup=true
else
    needs_sqlite=false
fi

if [ -z "$CODEX_REASONING_EFFORT" ]; then
    needs_codex_reasoning=true
    needs_setup=true
else
    needs_codex_reasoning=false
fi

if [ -z "$FAKE_REASONING_MAX_TOKENS" ]; then
    needs_fake_reasoning_tokens=true
    needs_setup=true
else
    needs_fake_reasoning_tokens=false
fi

# Interactive setup for missing values only
if [ "$needs_setup" = true ]; then
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}║     Open AI Gateway - Setup Inicial      ║${RESET}"
    echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
    echo ""

    if [ -f .env ]; then
        echo -e "${DIM}Arquivo .env encontrado, mas faltam algumas configurações.${RESET}"
        echo ""
    fi

    # 1. Proxy password
    if [ "$needs_proxy_key" = true ]; then
        echo -e "${CYAN}→${RESET} ${BOLD}Senha do proxy${RESET}"
        echo -e "${DIM}Essa senha protege seu gateway. Você vai usá-la como api_key nos clientes.${RESET}"
        echo -e "${DIM}Pode ser qualquer coisa que você inventar (ex: minha-senha-secreta-123)${RESET}"
        echo ""
        read -rp "   Senha: " PROXY_API_KEY
        echo ""

        if [ -z "$PROXY_API_KEY" ]; then
            echo "Erro: senha não pode ser vazia."
            exit 1
        fi
    fi

    # 2. Host
    if [ "$needs_host" = true ]; then
        echo -e "${CYAN}→${RESET} ${BOLD}Onde o proxy vai escutar?${RESET}"
        echo -e "${DIM}127.0.0.1 = só conexões locais (mais seguro)${RESET}"
        echo -e "${DIM}0.0.0.0   = aceita conexões de qualquer lugar (rede local, VPN, etc)${RESET}"
        echo ""
        read -rp "   Host [127.0.0.1]: " SERVER_HOST
        SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
        echo ""
    fi

    # 3. SQLite path
    if [ "$needs_sqlite" = true ]; then
        echo -e "${CYAN}→${RESET} ${BOLD}Caminho do banco SQLite3 do kiro-cli${RESET}"
        echo -e "${DIM}Locais comuns:${RESET}"
        echo -e "${DIM}  Linux/macOS: ~/.local/share/kiro-cli/data.sqlite3${RESET}"
        echo -e "${DIM}  Alternativo: ~/.local/share/amazon-q/data.sqlite3${RESET}"
        echo ""
        echo -e "${YELLOW}Não sabe onde fica? Mande o seguinte prompt para uma IA:${RESET}"
        echo ""
        echo -e "  ${DIM}\"Preciso encontrar o arquivo data.sqlite3 do kiro-cli ou amazon-q"
        echo -e "  no meu sistema. Rode 'find ~ -name data.sqlite3 -path \"*kiro*\" -o"
        echo -e "  -name data.sqlite3 -path \"*amazon-q*\" 2>/dev/null' e me diga o caminho.\"${RESET}"
        echo ""
        read -rp "   Caminho do SQLite3: " KIRO_CLI_DB_FILE
        echo ""

        if [ -z "$KIRO_CLI_DB_FILE" ]; then
            echo "Erro: caminho do SQLite3 não pode ser vazio."
            exit 1
        fi

        # Expand ~ for validation
        expanded_path="${KIRO_CLI_DB_FILE/#\~/$HOME}"
        if [ ! -f "$expanded_path" ]; then
            echo -e "${YELLOW}Aviso: arquivo não encontrado em '$expanded_path'${RESET}"
            read -rp "   Continuar mesmo assim? [s/N]: " confirm
            if [[ ! "$confirm" =~ ^[sS]$ ]]; then
                exit 1
            fi
            echo ""
        fi
    fi

    # 4. Codex reasoning effort
    if [ "$needs_codex_reasoning" = true ]; then
        echo -e "${CYAN}→${RESET} ${BOLD}Nível de raciocínio do Codex${RESET}"
        echo -e "${DIM}Controla o esforço de raciocínio para modelos gpt-*/codex-* roteados pelo Codex Provider.${RESET}"
        echo -e "${DIM}  concise  = respostas mais rápidas e diretas${RESET}"
        echo -e "${DIM}  auto     = equilíbrio automático (recomendado)${RESET}"
        echo -e "${DIM}  detailed = raciocínio mais profundo, respostas mais lentas${RESET}"
        echo ""
        read -rp "   Nível [auto]: " CODEX_REASONING_EFFORT
        CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-auto}"
        echo ""
    fi

    # 5. Fake reasoning max tokens
    if [ "$needs_fake_reasoning_tokens" = true ]; then
        echo -e "${CYAN}→${RESET} ${BOLD}Máximo de tokens para o Fake Reasoning${RESET}"
        echo -e "${DIM}Orçamento padrão de tokens de raciocínio quando o cliente não especifica.${RESET}"
        echo -e "${DIM}Valores maiores permitem raciocínio mais detalhado, mas aumentam o tempo de resposta.${RESET}"
        echo -e "${DIM}Recomendado: 4000 (padrão) a 10000 (máximo)${RESET}"
        echo ""
        read -rp "   Max tokens [4000]: " FAKE_REASONING_MAX_TOKENS
        FAKE_REASONING_MAX_TOKENS="${FAKE_REASONING_MAX_TOKENS:-4000}"
        echo ""
    fi

    # 6. OpenRouter (opcional)
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo -e "${CYAN}→${RESET} ${BOLD}OpenRouter (opcional)${RESET}"
        echo -e "${DIM}Permite acessar centenas de modelos (GPT-4o, Llama, Gemini, DeepSeek, etc.)${RESET}"
        echo -e "${DIM}através de uma única API. Obtenha sua chave em: https://openrouter.ai/keys${RESET}"
        echo ""
        read -rp "   Deseja configurar o OpenRouter? [s/N]: " configure_openrouter
        if [[ "$configure_openrouter" =~ ^[sS]$ ]]; then
            read -rp "   API Key do OpenRouter: " OPENROUTER_API_KEY
            echo ""
        else
            OPENROUTER_API_KEY=""
        fi
    fi

    # Write .env preserving any extra vars the user may have added manually
    if [ -f .env ]; then
        # Update existing .env: replace or append each var
        _update_env_var() {
            local key="$1" value="$2"
            if grep -q "^${key}=" .env 2>/dev/null; then
                sed -i "s|^${key}=.*|${key}=\"${value}\"|" .env
            elif grep -q "^# *${key}=" .env 2>/dev/null; then
                sed -i "s|^# *${key}=.*|${key}=\"${value}\"|" .env
            else
                echo "${key}=\"${value}\"" >> .env
            fi
        }

        [ "$needs_proxy_key" = true ] && _update_env_var "PROXY_API_KEY" "$PROXY_API_KEY"
        [ "$needs_host" = true ] && _update_env_var "SERVER_HOST" "$SERVER_HOST"
        [ "$needs_sqlite" = true ] && _update_env_var "KIRO_CLI_DB_FILE" "$KIRO_CLI_DB_FILE"
        [ "$needs_codex_reasoning" = true ] && _update_env_var "CODEX_REASONING_EFFORT" "$CODEX_REASONING_EFFORT"
        [ "$needs_fake_reasoning_tokens" = true ] && _update_env_var "FAKE_REASONING_MAX_TOKENS" "$FAKE_REASONING_MAX_TOKENS"
        [ -n "$OPENROUTER_API_KEY" ] && _update_env_var "OPENROUTER_API_KEY" "$OPENROUTER_API_KEY"
    else
        # Create fresh .env
        cat > .env << EOF
# Gerado por start.sh em $(date '+%Y-%m-%d %H:%M:%S')

PROXY_API_KEY="$PROXY_API_KEY"
SERVER_HOST="$SERVER_HOST"
KIRO_CLI_DB_FILE="$KIRO_CLI_DB_FILE"
CODEX_REASONING_EFFORT="$CODEX_REASONING_EFFORT"
FAKE_REASONING_MAX_TOKENS="$FAKE_REASONING_MAX_TOKENS"
OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
EOF
    fi

    echo -e "${GREEN}✓ Configuração salva no .env!${RESET}"
    echo ""

    # 6. Configure Claude Code
    echo -e "${CYAN}→${RESET} ${BOLD}Configurar o Claude Code agora?${RESET}"
    echo -e "${DIM}Isso vai adicionar no topo do ~/.claude/settings.json:${RESET}"
    echo -e "${DIM}  {${RESET}"
    echo -e "${DIM}    \"env\": {${RESET}"
    echo -e "${DIM}      \"ANTHROPIC_AUTH_TOKEN\": \"$PROXY_API_KEY\",${RESET}"
    echo -e "${DIM}      \"ANTHROPIC_MODEL\": \"claude-sonnet-4-6\",${RESET}"
    echo -e "${DIM}      \"ANTHROPIC_BASE_URL\": \"http://${SERVER_HOST}:8000\"${RESET}"
    echo -e "${DIM}    },${RESET}"
    echo -e "${DIM}    ...${RESET}"
    echo -e "${DIM}  }${RESET}"
    echo ""
    read -rp "   Configurar? [s/N]: " configure_claude
    echo ""
    if [[ "$configure_claude" =~ ^[sS]$ ]]; then
        SETTINGS_FILE="$HOME/.claude/settings.json"
        mkdir -p "$HOME/.claude"

        if [ -f "$SETTINGS_FILE" ]; then
            # Merge: inject/overwrite the "env" key at the top level using Python
            python3 - "$SETTINGS_FILE" "$PROXY_API_KEY" "${SERVER_HOST}:8000" << 'PYEOF'
import sys, json, collections

settings_path = sys.argv[1]
auth_token    = sys.argv[2]
base_url      = "http://" + sys.argv[3]

with open(settings_path, "r") as f:
    data = json.load(f, object_pairs_hook=collections.OrderedDict)

env_block = collections.OrderedDict([
    ("ANTHROPIC_AUTH_TOKEN", auth_token),
    ("ANTHROPIC_MODEL",      "claude-sonnet-4-6"),
    ("ANTHROPIC_BASE_URL",   base_url),
])

# Merge with any existing env keys (new keys win)
existing_env = data.pop("env", collections.OrderedDict())
existing_env.update(env_block)

# Rebuild with env first
new_data = collections.OrderedDict([("env", existing_env)])
new_data.update(data)

with open(settings_path, "w") as f:
    json.dump(new_data, f, indent=2)
    f.write("\n")
PYEOF
        else
            # Create fresh settings.json
            cat > "$SETTINGS_FILE" << EOF
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "$PROXY_API_KEY",
    "ANTHROPIC_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_BASE_URL": "http://${SERVER_HOST}:8000"
  }
}
EOF
        fi

        echo -e "${GREEN}✓ Claude Code configurado em $SETTINGS_FILE!${RESET}"
        echo ""
    fi
fi

