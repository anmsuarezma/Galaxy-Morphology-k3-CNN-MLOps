"""Tests de métricas con arreglos pequeños cuyo resultado se calcula a mano.

Ejemplo principal (K = 3)::

    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]

                 pred 0  pred 1  pred 2   | soporte (suma fila)
    real 0         2       1       0      |   3
    real 1         0       1       1      |   2
    real 2         0       0       1      |   1
    predichas      2       2       2        (suma columna)

    accuracy  = (2 + 1 + 1) / 6           = 2/3
    precision = [2/2, 1/2, 1/2]           = [1, 0.5, 0.5]
    recall    = [2/3, 1/2, 1/1]           = [2/3, 0.5, 1]
    F1        = [2*1*(2/3)/(5/3), 0.5, 2*0.5*1/1.5] = [0.8, 0.5, 2/3]
    macro F1  = (0.8 + 0.5 + 2/3) / 3     = 59/90
"""

import math

import pytest
import torch
from sklearn import metrics as skm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from galaxy3.metrics import (
    accuracy,
    classification_report,
    collect_predictions,
    confusion_matrix,
    generalization_gap,
    macro_f1,
    per_class_metrics,
)
from galaxy3.models import GalaxyCNN

CPU = torch.device("cpu")
Y_TRUE = torch.tensor([0, 0, 0, 1, 1, 2])
Y_PRED = torch.tensor([0, 0, 1, 1, 2, 2])
EXPECTED_CM = torch.tensor([[2, 1, 0], [0, 1, 1], [0, 0, 1]])


def _approx(values: torch.Tensor, expected: list[float]) -> None:
    assert values.tolist() == pytest.approx(expected)


def test_confusion_matrix_by_hand():
    cm = confusion_matrix(Y_TRUE, Y_PRED, num_classes=3)
    assert cm.dtype == torch.long
    assert torch.equal(cm, EXPECTED_CM)
    assert cm.sum().item() == len(Y_TRUE)


def test_confusion_matrix_keeps_size_for_absent_classes():
    cm = confusion_matrix(torch.tensor([0, 1]), torch.tensor([0, 1]), num_classes=4)
    assert cm.shape == (4, 4)
    assert torch.equal(cm.diagonal(), torch.tensor([1, 1, 0, 0]))


def test_accuracy_by_hand():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert accuracy(Y_TRUE, Y_TRUE) == 1.0


def test_per_class_metrics_by_hand():
    precision, recall, f1 = per_class_metrics(EXPECTED_CM)
    _approx(precision, [1.0, 0.5, 0.5])
    _approx(recall, [2 / 3, 0.5, 1.0])
    _approx(f1, [0.8, 0.5, 2 / 3])


def test_macro_f1_by_hand():
    assert macro_f1(EXPECTED_CM) == pytest.approx(59 / 90)


def test_zero_division_gives_zero():
    # Clase 1: nunca predicha -> precision 0, recall 0/1 = 0.
    # Clase 2: sin muestras ni predicciones -> todo 0.
    # Clase 0: P = 2/3, R = 2/2 = 1, F1 = 2*(2/3)*1/(5/3) = 0.8.
    cm = confusion_matrix(torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]), num_classes=3)
    precision, recall, f1 = per_class_metrics(cm)
    _approx(precision, [2 / 3, 0.0, 0.0])
    _approx(recall, [1.0, 0.0, 0.0])
    _approx(f1, [0.8, 0.0, 0.0])
    assert macro_f1(cm) == pytest.approx(0.8 / 3)
    assert all(math.isfinite(v) for v in f1.tolist())


def test_classification_report_by_hand():
    report = classification_report(Y_TRUE, Y_PRED, num_classes=3)
    assert report.accuracy == pytest.approx(2 / 3)
    assert torch.equal(report.confusion_matrix, EXPECTED_CM)
    _approx(report.precision, [1.0, 0.5, 0.5])
    _approx(report.recall, [2 / 3, 0.5, 1.0])
    _approx(report.f1, [0.8, 0.5, 2 / 3])
    assert report.macro_f1 == pytest.approx(59 / 90)


def test_classification_report_matches_sklearn_on_random_labels():
    g = torch.Generator().manual_seed(0)
    y_true = torch.randint(0, 3, (200,), generator=g)
    y_pred = torch.randint(0, 3, (200,), generator=g)
    report = classification_report(y_true, y_pred, num_classes=3)

    labels = [0, 1, 2]
    assert report.confusion_matrix.tolist() == skm.confusion_matrix(y_true, y_pred).tolist()
    precision, recall, f1, _ = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    _approx(report.precision, precision.tolist())
    _approx(report.recall, recall.tolist())
    _approx(report.f1, f1.tolist())
    assert report.macro_f1 == pytest.approx(skm.f1_score(y_true, y_pred, average="macro"))
    assert report.accuracy == pytest.approx(skm.accuracy_score(y_true, y_pred))


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        (torch.tensor([0, 1]), torch.tensor([0, 1, 2]), "difieren"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), "entero 1D"),
        (torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), "entero 1D"),
        (torch.tensor([0, 3]), torch.tensor([0, 1]), "fuera de"),
        (torch.tensor([0, 1]), torch.tensor([-1, 1]), "fuera de"),
        (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), "vacíos"),
    ],
)
def test_invalid_labels_raise(y_true, y_pred, match):
    with pytest.raises(ValueError, match=match):
        confusion_matrix(y_true, y_pred, num_classes=3)


def test_invalid_confusion_matrix_raises():
    with pytest.raises(ValueError, match="cuadrada"):
        per_class_metrics(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="negativas"):
        per_class_metrics(torch.tensor([[1, -1], [0, 1]]))


def test_generalization_gap_by_hand():
    assert generalization_gap(train_acc_eval=0.95, val_acc=0.85) == pytest.approx(0.10)
    assert generalization_gap(train_acc_eval=0.80, val_acc=0.85) == pytest.approx(-0.05)


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan])
def test_generalization_gap_rejects_invalid_accuracy(bad):
    with pytest.raises(ValueError, match="train_acc_eval"):
        generalization_gap(train_acc_eval=bad, val_acc=0.5)
    with pytest.raises(ValueError, match="val_acc"):
        generalization_gap(train_acc_eval=0.5, val_acc=bad)


def test_collect_predictions_by_hand():
    # Modelo identidad: los logits son la entrada aplanada, así que se conocen de antemano.
    logits = torch.tensor(
        [
            [2.0, 0.0, 0.0],  # pred 0, real 0: acierto
            [0.0, 0.0, 0.0],  # empate -> argmax toma el primero (0), real 1: error
            [0.0, 0.0, 3.0],  # pred 2, real 2: acierto
            [0.0, 1.0, 0.0],  # pred 1, real 0: error
            [0.0, 5.0, 0.0],  # pred 1, real 1: acierto
        ]
    )
    y = torch.tensor([0, 1, 2, 0, 1])
    loader = DataLoader(TensorDataset(logits.view(5, 3, 1, 1), y), batch_size=2)  # último: 1
    model = nn.Flatten()

    predictions = collect_predictions(model, loader, nn.CrossEntropyLoss(), CPU)

    assert torch.equal(predictions.y_true, y)
    assert torch.equal(predictions.y_pred, torch.tensor([0, 0, 2, 1, 1]))
    assert predictions.accuracy == pytest.approx(3 / 5)
    _approx(predictions.probabilities[1], [1 / 3, 1 / 3, 1 / 3])
    p0 = math.exp(2) / (math.exp(2) + 2)
    assert predictions.probabilities[0, 0].item() == pytest.approx(p0)
    assert predictions.loss == pytest.approx(nn.functional.cross_entropy(logits, y).item())


def test_collect_predictions_matches_direct_forward_and_leaves_model_in_eval():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(10, 3, 32, 32, generator=g)
    y = torch.randint(0, 3, (10,), generator=g)
    torch.manual_seed(0)
    model = GalaxyCNN(num_classes=3, use_batchnorm=True, dropout_p=0.3)
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    predictions = collect_predictions(model, loader, nn.CrossEntropyLoss(), CPU)

    assert not model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with torch.inference_mode():
        expected_logits = model(X)
    assert torch.equal(predictions.y_pred, expected_logits.argmax(dim=1))
    assert torch.allclose(predictions.probabilities, expected_logits.softmax(dim=1))
    assert predictions.probabilities.sum(dim=1).tolist() == pytest.approx([1.0] * 10)
    assert predictions.probabilities.shape == (10, 3)
    for tensor in (predictions.y_true, predictions.y_pred, predictions.probabilities):
        assert tensor.device == CPU
        assert not tensor.requires_grad

    report = classification_report(predictions.y_true, predictions.y_pred, num_classes=3)
    assert report.accuracy == pytest.approx(predictions.accuracy)
