"""
LLM Provider Configuration Commands.

Module: kautilya/commands/llm.py

Uses LLM adapter factory for provider-agnostic configuration and testing.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import click
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add adapters to path for unified configuration
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import adapter factory for LLM testing
try:
    from adapters.llm import create_sync_adapter, get_llm_config as get_adapter_config
    _ADAPTERS_AVAILABLE = True
except ImportError:
    _ADAPTERS_AVAILABLE = False

from ..config import load_llm_config, save_llm_config

console = Console()


@click.group(name="llm")
def llm_cmd() -> None:
    """Manage LLM provider configuration."""
    pass


@llm_cmd.command(name="config")
@click.option("--provider", help="Provider name (anthropic/openai/azure/local)")
@click.option("--model", help="Default model")
@click.option("--api-key-env", help="API key environment variable")
@click.pass_context
def llm_config_cmd(
    ctx: click.Context,
    provider: Optional[str],
    model: Optional[str],
    api_key_env: Optional[str],
) -> None:
    """Configure LLM provider credentials and models."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    configure_llm(config_dir, provider, model, api_key_env, interactive=False)


@llm_cmd.command(name="list")
def llm_list_cmd() -> None:
    """List available LLM adapters."""
    list_llm_providers()


@llm_cmd.command(name="test")
@click.pass_context
def llm_test_cmd(ctx: click.Context) -> None:
    """Test LLM connection."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    test_llm_connection(config_dir)


@llm_cmd.command(name="set-params")
@click.option("--provider", help="Provider name (defaults to current default)")
@click.option("--temperature", type=float, help="Temperature (0.0-2.0)")
@click.option("--max-tokens", type=int, help="Maximum tokens to generate")
@click.option("--top-p", type=float, help="Top-p sampling (0.0-1.0)")
@click.option("--top-k", type=int, help="Top-k sampling")
@click.option("--frequency-penalty", type=float, help="Frequency penalty (-2.0 to 2.0)")
@click.option("--presence-penalty", type=float, help="Presence penalty (-2.0 to 2.0)")
@click.option("--max-retries", type=int, help="Max API retries (0-10)")
@click.pass_context
def llm_set_params_cmd(
    ctx: click.Context,
    provider: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    top_k: Optional[int],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    max_retries: Optional[int],
) -> None:
    """Set hyperparameters for LLM provider."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    set_hyperparameters(
        config_dir,
        provider,
        temperature,
        max_tokens,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        max_retries,
        interactive=False,
    )


@llm_cmd.command(name="show-params")
@click.option("--provider", help="Provider name (defaults to current default)")
@click.pass_context
def llm_show_params_cmd(ctx: click.Context, provider: Optional[str]) -> None:
    """Show hyperparameters for LLM provider."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    show_hyperparameters(config_dir, provider)


def configure_llm(
    config_dir: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key_env: Optional[str] = None,
    interactive: bool = True,
) -> None:
    """
    Configure LLM provider.

    Args:
        config_dir: Configuration directory
        provider: Provider name
        model: Default model
        api_key_env: API key environment variable
        interactive: Use interactive prompts
    """
    console.print("\n[bold cyan]Configuring LLM Provider[/bold cyan]\n")

    # Get provider
    if not provider and interactive:
        provider = questionary.select(
            "Select provider:",
            choices=["anthropic", "openai", "azure", "gemini", "local", "vllm"],
        ).ask()
    elif not provider:
        provider = "anthropic"

    # Get model
    if not model and interactive:
        model = _select_model_for_provider(provider)
    elif not model:
        model = _get_default_model(provider)

    # Get API key source
    if interactive:
        api_key_source = questionary.select(
            "API key source:",
            choices=["env", "file", "vault"],
        ).ask()

        if api_key_source == "env":
            if not api_key_env:
                api_key_env = questionary.text(
                    "Environment variable:",
                    default=_get_default_api_key_env(provider),
                ).ask()
        else:
            console.print("[yellow]File and Vault sources not yet implemented, using env[/yellow]")
            api_key_env = _get_default_api_key_env(provider)
    else:
        api_key_env = api_key_env or _get_default_api_key_env(provider)

    # Set as default
    set_as_default = True
    if interactive:
        set_as_default = questionary.confirm("Set as default?", default=True).ask()

    # Load existing config
    llm_config = load_llm_config(config_dir)

    # Update config
    if "providers" not in llm_config:
        llm_config["providers"] = {}

    llm_config["providers"][provider] = {
        "default_model": model,
        "api_key_env": api_key_env,
        "fallback_model": _get_fallback_model(provider, model),
    }

    # Get endpoint for providers that need it
    if provider == "local":
        endpoint = questionary.text(
            "Ollama endpoint:", default="http://localhost:11434"
        ).ask() if interactive else "http://localhost:11434"
        llm_config["providers"][provider]["endpoint"] = endpoint

    elif provider == "vllm":
        endpoint = questionary.text(
            "vLLM endpoint:", default="http://localhost:8000"
        ).ask() if interactive else "http://localhost:8000"
        llm_config["providers"][provider]["endpoint"] = endpoint

    elif provider == "azure":
        endpoint = questionary.text(
            "Azure endpoint:", default="https://your-resource.openai.azure.com/"
        ).ask() if interactive else "https://your-resource.openai.azure.com/"
        llm_config["providers"][provider]["endpoint"] = endpoint

    elif provider == "gemini":
        # Gemini uses standard API endpoint, no need to configure
        pass

    if set_as_default:
        llm_config["default_provider"] = provider

    # Save config
    save_llm_config(llm_config, config_dir)

    # Show success
    success_message = f"""
[green]✓[/green] LLM config saved to {config_dir}/llm.yaml

[bold]Configuration:[/bold]
  Provider: {provider}
  Model: {model}
  API Key: ${api_key_env}
  Default: {set_as_default}
    """

    console.print(Panel(success_message.strip(), title="[bold green]LLM Configured[/bold green]"))


def list_llm_providers() -> None:
    """List available LLM providers."""
    table = Table(title="Available LLM Providers", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Default Model")
    table.add_column("Fallback Model")
    table.add_column("API Key Env")
    table.add_column("Endpoint", style="dim")

    providers = [
        ("anthropic", "claude-sonnet-4-20250514", "claude-haiku-4-20250514", "ANTHROPIC_API_KEY", "API"),
        ("openai", "gpt-4o", "gpt-4o-mini", "OPENAI_API_KEY", "API"),
        ("azure", "gpt-4o", "gpt-4o-mini", "AZURE_OPENAI_KEY", "Custom"),
        ("gemini", "gemini-2.0-flash", "gemini-1.5-pro", "GEMINI_API_KEY", "API"),
        ("local", "llama3.1:70b", "llama3.1:8b", "N/A", "localhost:11434"),
        ("vllm", "meta-llama/Llama-2-70b", "meta-llama/Llama-2-13b", "N/A", "localhost:8000"),
    ]

    for provider, default_model, fallback_model, api_key, endpoint in providers:
        table.add_row(provider, default_model, fallback_model, api_key, endpoint)

    console.print(table)
    console.print()

    # Add provider descriptions
    descriptions = [
        ("anthropic", "Anthropic Claude API", "✅ Production"),
        ("openai", "OpenAI GPT models", "✅ Production"),
        ("azure", "Azure OpenAI services", "✅ Production"),
        ("gemini", "Google Gemini API", "✅ Production"),
        ("local", "Ollama (local inference)", "⚠️ Requires server"),
        ("vllm", "vLLM (optimized local inference)", "⚠️ Requires server"),
    ]

    console.print("[bold]Provider Details:[/bold]")
    for provider, description, status in descriptions:
        console.print(f"  {provider:12} - {description:40} {status}")


def test_llm_connection(config_dir: str) -> None:
    """
    Test LLM connection.

    Checks configuration from multiple sources:
    1. .kautilya/llm.yaml config file
    2. .env file (OPENAI_API_KEY, OPENAI_MODEL)
    3. Environment variables

    Args:
        config_dir: Configuration directory
    """
    import os

    console.print("\n[bold cyan]Testing LLM Connection[/bold cyan]\n")

    llm_config = load_llm_config(config_dir)

    # Check if we have providers in config file
    if llm_config and "providers" in llm_config:
        default_provider = llm_config.get("default_provider", "anthropic")
        provider_config = llm_config["providers"].get(default_provider)

        if provider_config:
            api_key_env = provider_config.get("api_key_env")
            api_key = os.getenv(api_key_env, "")

            if not api_key and default_provider != "local":
                console.print(
                    f"[red]✗[/red] API key not found in environment variable ${api_key_env}"
                )
                return

            console.print(f"[green]✓[/green] Config Source: {config_dir}/llm.yaml")
            console.print(f"[green]✓[/green] Provider: {default_provider}")
            console.print(f"[green]✓[/green] Model: {provider_config.get('default_model')}")
            console.print(f"[green]✓[/green] API Key: ${api_key_env} {'(set)' if api_key else '(not set)'}")
            console.print("\n[dim]Note: Full connection test not yet implemented[/dim]")
            return

    # Try using adapter factory for unified configuration
    if _ADAPTERS_AVAILABLE:
        try:
            adapter_config = get_adapter_config()
            console.print(f"[green]✓[/green] Config Source: .env (via adapter factory)")
            console.print(f"[green]✓[/green] Provider: {adapter_config.get('provider', 'unknown')}")
            console.print(f"[green]✓[/green] Model: {adapter_config.get('model', 'unknown')}")
            console.print(f"[green]✓[/green] API Key: {'set' if adapter_config.get('api_key_set') else 'not set'}")

            # Try to test the connection using the adapter
            try:
                adapter = create_sync_adapter()
                response = adapter.complete_text("Hi", max_tokens=5)
                console.print(f"[green]✓[/green] Connection: Successfully connected!")
                console.print(f"[dim]  Response: {response[:50]}...[/dim]" if len(response) > 50 else f"[dim]  Response: {response}[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Connection test failed: {e}")

            console.print("\n[dim]Tip: Run '/llm config' to create a config file with more options.[/dim]")
            return
        except Exception:
            pass  # Fall through to legacy OpenAI check

    # Fallback: Check .env file for OpenAI configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if openai_key:
        console.print(f"[green]✓[/green] Config Source: .env file")
        console.print(f"[green]✓[/green] Provider: openai")
        console.print(f"[green]✓[/green] Model: {openai_model}")
        console.print(f"[green]✓[/green] API Key: $OPENAI_API_KEY (set, {len(openai_key)} chars)")

        # Try to actually test the connection
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            # Quick test with minimal tokens
            # Use max_completion_tokens for reasoning models (o1, o3, gpt-5)
            is_reasoning_model = any(
                openai_model.startswith(prefix)
                for prefix in ("o1", "o3", "o4", "gpt-5")
            )
            if is_reasoning_model:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_completion_tokens=10,
                )
            else:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
            console.print(f"[green]✓[/green] Connection: Successfully connected!")
            console.print(f"[dim]  Response: {response.choices[0].message.content}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Connection test failed: {e}")

        console.print("\n[dim]Tip: Run '/llm config' to create a config file with more options.[/dim]")
        return

    # No configuration found
    console.print("[yellow]No LLM configuration found.[/yellow]")
    console.print("\n[bold]To configure LLM, either:[/bold]")
    console.print("  1. Set OPENAI_API_KEY in your .env file")
    console.print("  2. Run '/llm config' to configure providers")
    console.print("\n[dim]Example .env:[/dim]")
    console.print("  OPENAI_API_KEY=sk-...")
    console.print("  OPENAI_MODEL=gpt-4o-mini")


def _select_model_for_provider(provider: str) -> str:
    """Interactively select model for provider."""
    models_by_provider = {
        "anthropic": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-haiku-4-20250514",
        ],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "azure": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "local": ["llama3.1:70b", "llama3.1:8b", "mistral:latest"],
        "vllm": ["meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-13b-hf", "mistralai/Mistral-7B-v0.1"],
    }

    models = models_by_provider.get(provider, ["default"])
    return questionary.select(f"Select model for {provider}:", choices=models).ask()


def _get_default_model(provider: str) -> str:
    """Get default model for provider."""
    defaults = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "azure": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "local": "llama3.1:70b",
        "vllm": "meta-llama/Llama-2-70b-hf",
    }
    return defaults.get(provider, "default")


def _get_fallback_model(provider: str, default_model: str) -> str:
    """Get fallback model for provider."""
    fallbacks = {
        "anthropic": "claude-haiku-4-20250514",
        "openai": "gpt-4o-mini",
        "azure": "gpt-4o-mini",
        "gemini": "gemini-1.5-pro",
        "local": "llama3.1:8b",
        "vllm": "meta-llama/Llama-2-13b-hf",
    }
    return fallbacks.get(provider, default_model)


def _get_default_api_key_env(provider: str) -> str:
    """Get default API key environment variable."""
    mapping = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_OPENAI_KEY",
        "gemini": "GEMINI_API_KEY",
        "local": "N/A",
        "vllm": "N/A",
    }
    return mapping.get(provider, "API_KEY")


# Provider-specific parameter support
PROVIDER_SUPPORTED_PARAMS = {
    "anthropic": {
        "temperature": True,
        "max_tokens": True,
        "top_p": True,
        "top_k": True,
        "frequency_penalty": False,
        "presence_penalty": False,
        "max_retries": True,
    },
    "openai": {
        "temperature": True,
        "max_tokens": True,
        "top_p": True,
        "top_k": False,
        "frequency_penalty": True,
        "presence_penalty": True,
        "max_retries": True,
    },
    "azure": {
        "temperature": True,
        "max_tokens": True,
        "top_p": True,
        "top_k": False,
        "frequency_penalty": True,
        "presence_penalty": True,
        "max_retries": True,
    },
    "gemini": {
        "temperature": True,
        "max_tokens": True,
        "top_p": True,
        "top_k": True,
        "frequency_penalty": False,
        "presence_penalty": False,
        "max_retries": True,
    },
    "local": {
        "temperature": True,
        "max_tokens": True,
        "top_p": True,
        "top_k": True,
        "frequency_penalty": False,
        "presence_penalty": False,
        "max_retries": True,
    },
    "vllm": {
        "temperature": True,
        "max_tokens": True,
        "top_p": True,
        "top_k": True,
        "frequency_penalty": False,
        "presence_penalty": False,
        "max_retries": True,
    },
}


def get_supported_params(provider: str) -> Dict[str, bool]:
    """
    Get supported parameters for a provider.

    Args:
        provider: Provider name

    Returns:
        Dictionary of parameter names to support status
    """
    return PROVIDER_SUPPORTED_PARAMS.get(provider, PROVIDER_SUPPORTED_PARAMS["openai"])


def validate_param_for_provider(provider: str, param_name: str) -> bool:
    """
    Check if a parameter is supported by a provider.

    Args:
        provider: Provider name
        param_name: Parameter name

    Returns:
        True if supported, False otherwise
    """
    supported_params = get_supported_params(provider)
    return supported_params.get(param_name, False)


def configure_llm_programmatic(
    provider: str,
    model: Optional[str] = None,
    set_default: bool = False,
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for LLM configuration.

    Args:
        provider: Provider name
        model: Model name
        set_default: Set as default provider
        config_dir: Configuration directory

    Returns:
        Configuration dictionary
    """
    model = model or _get_default_model(provider)
    api_key_env = _get_default_api_key_env(provider)

    llm_config = load_llm_config(config_dir)

    if "providers" not in llm_config:
        llm_config["providers"] = {}

    llm_config["providers"][provider] = {
        "default_model": model,
        "api_key_env": api_key_env,
        "fallback_model": _get_fallback_model(provider, model),
    }

    if set_default:
        llm_config["default_provider"] = provider

    save_llm_config(llm_config, config_dir)

    return {
        "provider": provider,
        "model": model,
        "api_key_env": api_key_env,
        "set_as_default": set_default,
    }


def test_llm_connection_programmatic(config_dir: str = ".kautilya") -> dict:
    """
    Programmatic interface for testing LLM connection.

    Args:
        config_dir: Configuration directory

    Returns:
        Test result dictionary
    """
    import os
    import time

    llm_config = load_llm_config(config_dir)

    if not llm_config or "providers" not in llm_config:
        raise ValueError("No LLM providers configured")

    default_provider = llm_config.get("default_provider", "openai")
    provider_config = llm_config["providers"].get(default_provider)

    if not provider_config:
        raise ValueError(f"Provider {default_provider} not configured")

    api_key_env = provider_config.get("api_key_env")
    api_key = os.getenv(api_key_env, "") if api_key_env != "N/A" else "local"

    start_time = time.time()

    # Simulated test (actual implementation would make API call)
    success = bool(api_key) or default_provider == "local"
    response_time_ms = int((time.time() - start_time) * 1000)

    return {
        "provider": default_provider,
        "model": provider_config.get("default_model"),
        "api_key_set": bool(api_key),
        "success": success,
        "response_time_ms": response_time_ms,
    }


def set_hyperparameters(
    config_dir: str,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    max_retries: Optional[int] = None,
    interactive: bool = True,
) -> None:
    """
    Set hyperparameters for LLM provider.

    Args:
        config_dir: Configuration directory
        provider: Provider name (None = use default)
        temperature: Temperature (0.0-2.0)
        max_tokens: Maximum tokens
        top_p: Top-p sampling (0.0-1.0)
        top_k: Top-k sampling
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        max_retries: Max retries (0-10)
        interactive: Use interactive prompts
    """
    console.print("\n[bold cyan]Configuring LLM Hyperparameters[/bold cyan]\n")

    # Load config
    llm_config = load_llm_config(config_dir)

    if not llm_config or "providers" not in llm_config:
        console.print("[yellow]No LLM providers configured. Run /llm config first.[/yellow]")
        return

    # Determine provider
    if not provider:
        provider = llm_config.get("default_provider", "anthropic")

    if provider not in llm_config["providers"]:
        console.print(f"[red]Provider {provider} not configured.[/red]")
        return

    provider_config = llm_config["providers"][provider]

    # Get supported parameters for this provider
    supported_params = get_supported_params(provider)

    # Get current hyperparameters or create defaults
    current_params = provider_config.get("hyperparameters", {})

    # Validate CLI parameters against provider support (non-interactive mode)
    if not interactive:
        warnings = []
        if temperature is not None and not supported_params.get("temperature"):
            warnings.append("temperature")
        if max_tokens is not None and not supported_params.get("max_tokens"):
            warnings.append("max_tokens")
        if top_p is not None and not supported_params.get("top_p"):
            warnings.append("top_p")
        if top_k is not None and not supported_params.get("top_k"):
            warnings.append("top_k")
        if frequency_penalty is not None and not supported_params.get("frequency_penalty"):
            warnings.append("frequency_penalty")
        if presence_penalty is not None and not supported_params.get("presence_penalty"):
            warnings.append("presence_penalty")

        if warnings:
            console.print(f"\n[yellow]Warning: {provider} does not support the following parameters:[/yellow]")
            for param in warnings:
                console.print(f"  - {param}")
            console.print("\n[dim]These parameters will be ignored.[/dim]\n")

    # Interactive mode
    if interactive:
        console.print(f"[bold]Configuring hyperparameters for: {provider}[/bold]\n")
        console.print("[dim]Only showing parameters supported by this provider[/dim]")
        console.print("[dim]Leave blank to keep current value[/dim]\n")

        # Temperature (supported by all providers)
        if supported_params.get("temperature"):
            current_temp = current_params.get("temperature", 0.7)
            temp_input = questionary.text(
                f"Temperature (0.0-2.0, current: {current_temp}):",
                default=str(current_temp) if temperature is None else str(temperature),
            ).ask()
            temperature = float(temp_input) if temp_input else current_temp

        # Max tokens (supported by all providers)
        if supported_params.get("max_tokens"):
            current_max_tokens = current_params.get("max_tokens")
            max_tokens_str = str(current_max_tokens) if current_max_tokens else "None"
            max_tokens_input = questionary.text(
                f"Max tokens (current: {max_tokens_str}):",
                default=str(max_tokens) if max_tokens else max_tokens_str,
            ).ask()
            if max_tokens_input and max_tokens_input.lower() != "none":
                max_tokens = int(max_tokens_input)
            else:
                max_tokens = None

        # Top-p (not supported by all providers)
        if supported_params.get("top_p"):
            current_top_p = current_params.get("top_p")
            top_p_str = str(current_top_p) if current_top_p else "None"
            top_p_input = questionary.text(
                f"Top-p (0.0-1.0, current: {top_p_str}):",
                default=str(top_p) if top_p else top_p_str,
            ).ask()
            if top_p_input and top_p_input.lower() != "none":
                top_p = float(top_p_input)
            else:
                top_p = None

        # Top-k (Anthropic, Gemini, Local only)
        if supported_params.get("top_k"):
            current_top_k = current_params.get("top_k")
            top_k_str = str(current_top_k) if current_top_k else "None"
            top_k_input = questionary.text(
                f"Top-k (current: {top_k_str}):",
                default=str(top_k) if top_k else top_k_str,
            ).ask()
            if top_k_input and top_k_input.lower() != "none":
                top_k = int(top_k_input)
            else:
                top_k = None

        # Frequency penalty (OpenAI, Azure only)
        if supported_params.get("frequency_penalty"):
            current_freq = current_params.get("frequency_penalty")
            freq_str = str(current_freq) if current_freq else "None"
            freq_input = questionary.text(
                f"Frequency penalty (-2.0 to 2.0, current: {freq_str}):",
                default=str(frequency_penalty) if frequency_penalty else freq_str,
            ).ask()
            if freq_input and freq_input.lower() != "none":
                frequency_penalty = float(freq_input)
            else:
                frequency_penalty = None

        # Presence penalty (OpenAI, Azure only)
        if supported_params.get("presence_penalty"):
            current_pres = current_params.get("presence_penalty")
            pres_str = str(current_pres) if current_pres else "None"
            pres_input = questionary.text(
                f"Presence penalty (-2.0 to 2.0, current: {pres_str}):",
                default=str(presence_penalty) if presence_penalty else pres_str,
            ).ask()
            if pres_input and pres_input.lower() != "none":
                presence_penalty = float(pres_input)
            else:
                presence_penalty = None

        # Max retries (supported by all providers)
        if supported_params.get("max_retries"):
            current_retries = current_params.get("max_retries", 3)
            retries_input = questionary.text(
                f"Max retries (0-10, current: {current_retries}):",
                default=str(max_retries) if max_retries else str(current_retries),
            ).ask()
            max_retries = int(retries_input) if retries_input else current_retries

    # Update hyperparameters - only save supported parameters
    hyperparams = {}

    # Temperature (always supported)
    if supported_params.get("temperature"):
        hyperparams["temperature"] = temperature if temperature is not None else current_params.get("temperature", 0.7)

    # Max tokens (always supported)
    if supported_params.get("max_tokens"):
        if max_tokens is not None:
            hyperparams["max_tokens"] = max_tokens
        elif current_params.get("max_tokens"):
            hyperparams["max_tokens"] = current_params["max_tokens"]

    # Top-p
    if supported_params.get("top_p") and top_p is not None:
        hyperparams["top_p"] = top_p
    elif not supported_params.get("top_p") and current_params.get("top_p"):
        # Remove unsupported parameter from config
        pass

    # Top-k
    if supported_params.get("top_k") and top_k is not None:
        hyperparams["top_k"] = top_k

    # Frequency penalty
    if supported_params.get("frequency_penalty") and frequency_penalty is not None:
        hyperparams["frequency_penalty"] = frequency_penalty

    # Presence penalty
    if supported_params.get("presence_penalty") and presence_penalty is not None:
        hyperparams["presence_penalty"] = presence_penalty

    # Max retries (always supported)
    if supported_params.get("max_retries"):
        hyperparams["max_retries"] = max_retries if max_retries is not None else current_params.get("max_retries", 3)

    # Save to config
    llm_config["providers"][provider]["hyperparameters"] = hyperparams
    save_llm_config(llm_config, config_dir)

    # Show success
    success_message = f"""
[green]✓[/green] Hyperparameters saved for {provider}

[bold]Configuration:[/bold]
  Temperature: {hyperparams.get('temperature', 0.7)}
  Max Tokens: {hyperparams.get('max_tokens', 'None')}
  Top-p: {hyperparams.get('top_p', 'None')}
  Top-k: {hyperparams.get('top_k', 'None')}
  Frequency Penalty: {hyperparams.get('frequency_penalty', 'None')}
  Presence Penalty: {hyperparams.get('presence_penalty', 'None')}
  Max Retries: {hyperparams.get('max_retries', 3)}
    """

    console.print(Panel(success_message.strip(), title="[bold green]Hyperparameters Configured[/bold green]"))


def show_hyperparameters(config_dir: str, provider: Optional[str] = None) -> None:
    """
    Show hyperparameters for LLM provider.

    Args:
        config_dir: Configuration directory
        provider: Provider name (None = use default)
    """
    # Load config
    llm_config = load_llm_config(config_dir)

    if not llm_config or "providers" not in llm_config:
        console.print("[yellow]No LLM providers configured. Run /llm config first.[/yellow]")
        return

    # Determine provider
    if not provider:
        provider = llm_config.get("default_provider", "anthropic")

    if provider not in llm_config["providers"]:
        console.print(f"[red]Provider {provider} not configured.[/red]")
        return

    provider_config = llm_config["providers"][provider]
    hyperparams = provider_config.get("hyperparameters", {})

    # Get supported parameters for this provider
    supported_params = get_supported_params(provider)

    # Create table
    table = Table(title=f"Hyperparameters for {provider}", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")
    table.add_column("Range", style="dim")
    table.add_column("Supported", justify="center")

    # Add rows with support indicators
    def add_param_row(param_name: str, value: Any, range_str: str) -> None:
        """Add parameter row with support indicator."""
        is_supported = supported_params.get(param_name, False)
        support_indicator = "[green]✓[/green]" if is_supported else "[red]✗[/red]"

        # Gray out unsupported parameters
        if not is_supported:
            param_style = "dim cyan"
            value_style = "dim"
        else:
            param_style = "cyan"
            value_style = ""

        table.add_row(
            f"[{param_style}]{param_name}[/{param_style}]",
            f"[{value_style}]{value}[/{value_style}]",
            f"[dim]{range_str}[/dim]",
            support_indicator
        )

    add_param_row("temperature", hyperparams.get("temperature", 0.7), "0.0 - 2.0")
    add_param_row("max_tokens", hyperparams.get("max_tokens", "None"), "> 0")
    add_param_row("top_p", hyperparams.get("top_p", "None"), "0.0 - 1.0")
    add_param_row("top_k", hyperparams.get("top_k", "None"), "> 0")
    add_param_row("frequency_penalty", hyperparams.get("frequency_penalty", "None"), "-2.0 - 2.0")
    add_param_row("presence_penalty", hyperparams.get("presence_penalty", "None"), "-2.0 - 2.0")
    add_param_row("max_retries", hyperparams.get("max_retries", 3), "0 - 10")

    console.print()
    console.print(table)
    console.print()

    # Add parameter descriptions
    console.print("[bold]Parameter Descriptions:[/bold]")
    console.print("  [cyan]temperature[/cyan]         - Controls randomness (higher = more random)")
    console.print("  [cyan]max_tokens[/cyan]          - Maximum tokens to generate")
    console.print("  [cyan]top_p[/cyan]               - Nucleus sampling (alternative to temperature)")
    console.print("  [cyan]top_k[/cyan]               - Top-k sampling (limits vocabulary)")
    console.print("  [cyan]frequency_penalty[/cyan]   - Reduces repetition based on frequency")
    console.print("  [cyan]presence_penalty[/cyan]    - Encourages topic diversity")
    console.print("  [cyan]max_retries[/cyan]         - Max API call retries on failure")
    console.print()

    # Add note about unsupported parameters
    unsupported = [param for param, supported in supported_params.items() if not supported]
    if unsupported:
        console.print(f"[dim]Note: {provider} does not support: {', '.join(unsupported)}[/dim]")
        console.print()
