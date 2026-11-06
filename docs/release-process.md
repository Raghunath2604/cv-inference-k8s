# Image release process

Every push to `main` runs linting, tests, training, Docker build, Compose validation, and Kubernetes schema validation. The workflow then builds the runtime image, scans it with Trivy for unfixed high and critical vulnerabilities, and publishes it to GitHub Container Registry.

Use the immutable commit tag for Kubernetes releases:

```text
ghcr.io/raghunath2604/cv-inference-k8s:sha-<full-commit-sha>
```

The `latest` tag is available for development only. Verify the deployed release with:

```bash
curl https://<service-host>/version
```

The response identifies both the application release and model version. Update `APP_VERSION` and `MODEL_VERSION` together with the image tag when promoting a new release.
