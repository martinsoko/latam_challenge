## Notes

### Environment issues

The commands to set up the environment failed (`make venv` and `make install`) so I had to change the pinned version of some dependencies. The same happened with `make model-test`.

### Unused variables

The DS created new features that were not used for model fitting, only for exploration. I'm not sure if this was on purpose or something they overlooked and I should've fixed.
I left it as is because of the instruction not to improve the model.

### Tests

There are issues with how the tests are written:
- `test_model_predict` depends on `test_model_fit` being called first -> I had to fix it, otherwise the test wouldn't pass.
- Tests are not really "unit" test. Test Methods call more than one target methods, so they're more like system or integration tests -> I dont't know if this was by design or should also be fixed. The fact that there are performance assertions in the tests leads me to think the original idea is to test the whole system and not code units, so I'll leave it as is.
- Tests expect a 400 status code when there's a validation error, but Pydantic returns 422 by default. To keep the tests unchanged I had to create a validation exception handler to change the status code instead.
- API test use `TestClient(app)` during setup, which doesn't work with `lifespan` context manager and `@app.on_event("startup"/"shutdown)` decorators. This means that model setup must be done at module level.

### Repo

- I chose to commit files I would not normally include in a repository, just to make the submission complete. models/ and data/


### Dockerfile

This dockerfile has the issue that the model is not persisted outside the container. If the container restarts, the trained model is lost and it will be re-trained on startup.
This is fine for the sake of the exercise, since it's not a really productive system. The data is small so it can be copied into the image and the model training is fast, so the startup delay is not a big deal.
In a truly production scenario, the data and the model asset(s) would be properly stored and versioned outside the container (for example, data in BigQuery and model assets in GCS) and only the relevant files would be loaded at startup.
Train and serve containers could have independent dockerfiles, each with it's own set of requirements and processes.

### CD smoke test

The CD workflow polls `/health` after a merge to `main` to verify the deployment, but the check will always pass because Cloud Run keeps the old revision serving until the new one is ready. The old version responds with 200 before the new one takes over, so the smoke test can't tell whether the new deployment actually succeeded.
To properly verify the new deployment, the `/health` endpoint should expose the `K_REVISION` environment variable (automatically injected by Cloud Run, unique per revision). The smoke test would then capture the current revision at workflow start and poll until it changes. I didn't want to make any changes to the endpoint because it could mess with your evaluation scripts.
