from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.config import (
    BASE_DIR,
    CV_DIR,
    CONFIG_FILE,
    CV_PROFILE_FILE,
    DATA_DIR,
    JobPreferences,
    Settings,
    load_config,
    save_config,
    load_preferences,
    save_preferences,
    load_cv_profile,
    save_cv_profile,
    settings,
)
from src.cv_parser import parse_cv_pdf
from src.database import init_db
from src.delivery import test_email_connection

console = Console()
logger = logging.getLogger(__name__)

WELCOME = """
# &#128269; Jobs Scout - Configuración Inicial

Te haré unas preguntas para personalizar la búsqueda de ofertas de trabajo.
El sistema buscará **diariamente** en LinkedIn, InfoJobs y Tecnoempleo,
y te enviará las 10 mejores ofertas por email cada mañana.

Pulsa `Ctrl+C` en cualquier momento para salir.
"""

EMPTY_STATE_ASCII = """
  &#9995; Aún no tengo un CV cargado.
  Coloca tu CV en PDF dentro de la carpeta `cv/` y vuelve a ejecutar `python src/setup.py`
"""


async def run_setup() -> None:
    console.clear()
    console.print(Markdown(WELCOME))

    console.print("\n[bold]Paso 1 de 3:[/bold] Configuración de email\n")

    config = load_config()

    email_user = Prompt.ask(
        "Email de Gmail para envío",
        default=config.get("email_user", settings.email_user or ""),
    )
    email_password = Prompt.ask(
        "App Password de Gmail (https://myaccount.google.com/apppasswords)",
        password=True,
        default=config.get("email_password", ""),
    )
    email_to = Prompt.ask(
        "Email donde recibir las ofertas",
        default=config.get("email_to", email_user),
    )

    config["email_user"] = email_user
    config["email_password"] = email_password
    config["email_to"] = email_to
    config["email_from"] = f"Jobs Scout <{email_user}>"

    send_hour = IntPrompt.ask(
        "Hora del día para recibir ofertas (formato 24h)",
        default=config.get("daily_send_hour", 9),
    )
    config["daily_send_hour"] = send_hour

    save_config(config)

    if email_user and email_password:
        with console.status("Probando conexión de email..."):
            os.environ["EMAIL_USER"] = email_user
            os.environ["EMAIL_PASSWORD"] = email_password
            os.environ["EMAIL_TO"] = email_to
            ok = await test_email_connection()
        if ok:
            console.print("[green]✓[/green] Conexión de email verificada")
        else:
            console.print("[yellow]⚠[/yellow] No se pudo verificar el email. Verifica las credenciales.")

    console.print("\n[bold]Paso 2 de 3:[/bold] Sube tu CV en PDF\n")

    cv_path = await _handle_cv_upload()

    console.print("\n[bold]Paso 3 de 3:[/bold] Preferencias de búsqueda\n")

    await _configure_preferences()

    console.print("\n[green bold]✓[/green bold] Configuración completada.\n")

    table = Table(title="Resumen de configuración")
    table.add_column("Campo", style="cyan")
    table.add_column("Valor", style="green")

    prefs = load_preferences()
    cv = load_cv_profile()
    conf = load_config()

    table.add_row("CV", cv.full_name if cv and cv.full_name else "No detectado")
    table.add_row("Cargo buscado", ", ".join(prefs.desired_titles[:3]))
    table.add_row("Tecnologías", ", ".join(prefs.tech_stack[:5]))
    table.add_row("Ubicación", prefs.location or "No especificada")
    table.add_row("Remoto", "Sí" if prefs.remote_only else "No requerido")
    table.add_row("Salario mínimo", f"{prefs.min_salary:,}€" if prefs.min_salary else "No especificado")
    table.add_row("Hora envío diario", f"{conf.get('daily_send_hour', 9)}:00")
    table.add_row("Email destino", conf.get("email_to", "No configurado"))

    console.print(table)

    console.print("\n[bold]Para iniciar el sistema:[/bold]")
    console.print("  python src/main.py")
    console.print("\n[bold]Para simular un escaneo manual:[/bold]")
    console.print("  python src/main.py --run-now")


async def _handle_cv_upload() -> Optional[Path]:
    cv_dir = CV_DIR
    existing_cv = list(cv_dir.glob("*.pdf"))

    if existing_cv:
        cv_path = existing_cv[0]
        console.print(f"CV encontrado: [cyan]{cv_path.name}[/cyan]")
        if Confirm.ask("¿Usar este CV?", default=True):
            with console.status("Analizando CV..."):
                try:
                    profile = await parse_cv_pdf(cv_path)
                    console.print(f"[green]✓[/green] CV analizado: {profile.full_name}")
                    console.print(f"    Tecnologías detectadas: {', '.join(profile.technologies[:10])}")
                    console.print(f"    Años de experiencia: {profile.experience_years}")
                    save_cv_profile(profile)
                    return cv_path
                except Exception as e:
                    console.print(f"[red]✗[/red] Error al parsear CV: {e}")
                    if Confirm.ask("¿Reintentar con otro archivo?", default=True):
                        pass
                    else:
                        return None

    while True:
        path_str = Prompt.ask(
            "Ruta del archivo PDF de tu CV (arrastra el archivo aquí o escribe la ruta)"
        )
        path_str = path_str.strip().strip('"').strip("'")
        cv_path = Path(path_str)

        if not cv_path.exists():
            console.print("[red]✗[/red] Archivo no encontrado. Intenta de nuevo.")
            continue
        if cv_path.suffix.lower() != ".pdf":
            console.print("[red]✗[/red] Solo se aceptan archivos PDF.")
            continue

        dest = cv_dir / cv_path.name
        shutil.copy2(cv_path, dest)
        console.print(f"[green]✓[/green] CV copiado a {dest}")

        with console.status("Analizando CV..."):
            try:
                profile = await parse_cv_pdf(dest)
                console.print(f"[green]✓[/green] CV analizado: {profile.full_name}")
                console.print(f"    Tecnologías detectadas: {', '.join(profile.technologies[:10])}")
                console.print(f"    Años de experiencia: {profile.experience_years}")
                save_cv_profile(profile)
                return dest
            except Exception as e:
                console.print(f"[red]✗[/red] Error al parsear CV: {e}")
                if Confirm.ask("¿Reintentar con otro archivo?", default=True):
                    continue
                return None


async def _configure_preferences() -> None:
    prefs = load_preferences()
    cv = load_cv_profile()

    console.print("Puedes escribir múltiples valores separados por comas.\n")

    titles = Prompt.ask(
        "¿Qué cargo(s) buscas? (ej: Backend Developer, Python Developer)",
        default=", ".join(prefs.desired_titles) if prefs.desired_titles else "",
    )
    prefs.desired_titles = [t.strip() for t in titles.split(",") if t.strip()]

    if cv and cv.technologies:
        console.print(f"Tecnologías detectadas en tu CV: [dim]{', '.join(cv.technologies[:10])}[/dim]")
        if Confirm.ask("¿Usar estas tecnologías para la búsqueda?", default=True):
            prefs.tech_stack = cv.technologies.copy()
        else:
            stack = Prompt.ask(
                "Tecnologías que quieres usar (separadas por comas)",
                default=", ".join(prefs.tech_stack) if prefs.tech_stack else "",
            )
            prefs.tech_stack = [t.strip() for t in stack.split(",") if t.strip()]
    else:
        stack = Prompt.ask(
            "Tecnologías / stack (separadas por comas)",
            default=", ".join(prefs.tech_stack) if prefs.tech_stack else "",
        )
        prefs.tech_stack = [t.strip() for t in stack.split(",") if t.strip()]

    prefs.location = Prompt.ask(
        "Ubicación (ciudad, país, o 'Remoto')",
        default=prefs.location or "Remoto España",
    )

    prefs.remote_only = Confirm.ask("¿Solo ofertas remotas?", default=prefs.remote_only)
    if not prefs.remote_only:
        prefs.hybrid_allowed = Confirm.ask("¿Permitir híbrido?", default=prefs.hybrid_allowed)
        prefs.onsite_allowed = Confirm.ask("¿Permitir presencial?", default=prefs.onsite_allowed)

    salary = Prompt.ask(
        "Salario mínimo anual en euros (0 = sin límite)",
        default=str(prefs.min_salary),
    )
    prefs.min_salary = int(salary) if salary.isdigit() else 0

    prefs.seniority = Prompt.ask(
        "Seniority (junior / mid / senior / lead / indiferente)",
        default=prefs.seniority or "indiferente",
    )

    exclude = Prompt.ask(
        "Palabras clave a EVITAR (separadas por comas, ej: WordPress, PHP)",
        default=", ".join(prefs.exclude_keywords) if prefs.exclude_keywords else "",
    )
    prefs.exclude_keywords = [t.strip() for t in exclude.split(",") if t.strip()]

    sectors = Prompt.ask(
        "Sectores a EVITAR (separados por comas, ej: consultancy, banking)",
        default=", ".join(prefs.exclude_sectors) if prefs.exclude_sectors else "",
    )
    prefs.exclude_sectors = [t.strip() for t in sectors.split(",") if t.strip()]

    prefs.languages = ["español nativo"]
    english = Prompt.ask("Nivel de inglés (A1/A2/B1/B2/C1/C2 / ninguno)", default="B2")
    if english.lower() != "ninguno":
        prefs.languages.append(f"inglés {english.upper()}")

    save_preferences(prefs)
    console.print("[green]✓[/green] Preferencias guardadas\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    init_db()
    asyncio.run(run_setup())
