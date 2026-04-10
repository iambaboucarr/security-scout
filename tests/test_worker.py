from worker import WorkerSettings, process_advisory_workflow_job


def test_worker_settings_registers_advisory_job() -> None:
    assert process_advisory_workflow_job in WorkerSettings.functions
    assert WorkerSettings.on_startup is not None
    assert WorkerSettings.on_shutdown is not None
