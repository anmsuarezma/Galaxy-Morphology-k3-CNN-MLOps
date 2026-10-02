# Galaxy3

El contrato del proyecto es `docs/SPEC_EXPERIMENTO_1.md`. Leerlo completo antes de
cualquier fase. Ante conflicto entre este archivo y la spec, preguntar.

## Reglas que nunca se rompen
- Trabajar UNA fase a la vez (sección 27). Al terminar: reportar, mostrar resultados de
  Ruff y pytest, y ESPERAR aprobación explícita. No encadenar fases.
- El test nunca entra en `fit()` ni en `run_experiment()`.
- El modelo devuelve logits. Un solo training loop.
- Sin scheduler, sin early stopping, `weight_decay=0.0`.
- Checkpoints cargables con `torch.load(..., weights_only=True)`.
- No implementar ResNet, Docker, APIs, AMP ni DDP.

## Comandos (desde la raíz del workspace)
- Tests: `uv run pytest -q galaxy3/tests`
- Lint: `uv run ruff check galaxy3 && uv run ruff format --check galaxy3`
