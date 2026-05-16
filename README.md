# Motor GMD-AAM: Economía de la Autonomía en Sistemas de IA Agéntica

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20219750.svg)](https://doi.org/10.5281/zenodo.20219750)

Este repositorio contiene el código fuente, los prechecks deterministas y el entorno de simulación utilizados para la validación metodológica de la tesis doctoral: *"Toma de decisiones empresariales en la era de la Cuarta Revolución Industrial: economía de la autonomía en sistemas de inteligencia artificial agéntica"*

La arquitectura implementa un **Autonomous Allocation Mechanism (AAM)** bajo el marco teórico de **Governance as Mechanism Design (GMD)**, orquestando sistemas multi-agente para la evaluación probabilística y legal de transacciones automatizadas

## Requisitos del Sistema
- Python 3.x
- [Ollama](https://ollama.com/) instalado y en ejecución
- Modelo local: `deepseek-r1:14b` (`ollama run deepseek-r1:14b`)

## Instalación y Replicación
1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/Bl1ndcat86/PhD-AAM-GMD.git](https://github.com/Bl1ndcat86/PhD-AAM-GMD.git)
   cd PhD-AAM-GMD
   
2. Crear el entorno virtual e instalar dependencias:
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

4. Ejecutar las simulaciones
#Ejemplo de ejecución del batch transaccional
python xxx_BATCH.py

## Estructura Principal del Repositorio
- /precheck/: Lógica de los invariantes contables y validación de expresiones regulares legales.
- /motors/ y /evaluation/: Lógica central del orquestador AAM y algoritmos de cálculo de la matriz de Pareto.
- /figures/: Scripts generadores de los gráficos y diagramas en formato APA 7 para el documento de tesis.

## Licencia y Citación
Este código se distribuye con fines de auditoría académica. Si utilizas esta arquitectura para replicación, por favor cita el identificador DOI adjunto en la cabecera.
