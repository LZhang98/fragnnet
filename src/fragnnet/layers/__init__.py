"""Shared torch building blocks (was ``model/``, renamed in reorganization Phase 8).

Only layers more than one model uses live here; whole models live in ``fragnnet.models`` and
spectrum losses in ``fragnnet.loss``. This initializer re-exports nothing, so importing one
module does not import the others; import from the defining module. Do not bind this package
or a module in it to the name ``nn``, which the code base reserves for ``torch.nn``.
"""
