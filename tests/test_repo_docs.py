from pathlib import Path


def test_required_readmes_exist():
    required = [
        Path('README.md'),
        Path('models/README.md'),
        Path('models_legacy_nc/README.md'),
        Path('data_prep/README.md'),
        Path('data_prep_legacy_nc/README.md'),
    ]
    for path in required:
        assert path.exists(), f"Missing documentation file: {path}"


def test_environment_file_exists_and_has_name():
    env_file = Path('environment.yml')
    assert env_file.exists(), 'Missing environment.yml'
    text = env_file.read_text()
    assert 'name: ailanthus-naip' in text


def test_root_readme_mentions_us_and_legacy():
    text = Path('README.md').read_text().lower()
    assert 'us-scale' in text
    assert 'legacy' in text
    assert 'models_legacy_nc' in text
