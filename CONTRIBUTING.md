# Contributing to poster2json

First off, thank you for considering contributing to poster2json! It's contributors like you that help make scientific poster metadata more accessible and FAIR.

## Why Read These Guidelines?

Following these guidelines helps communicate that you respect the time of the developers maintaining this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## What Contributions We're Looking For

poster2json is an open source project and we welcome contributions from the community! There are many ways to contribute:

- **Bug reports** - Found something that doesn't work? Let us know
- **Documentation improvements** - Clarify confusing sections, fix typos, add examples
- **Code contributions** - Bug fixes, new features, performance improvements
- **Testing** - Run the pipeline on your posters and report edge cases
- **Schema improvements** - Suggestions for the poster-json-schema

## What We're Not Looking For

- Please don't use the issue tracker for support questions. For help using poster2json, check our [documentation](docs/) first
- Large architectural changes without prior discussion - please open an issue first

## Ground Rules

**Responsibilities:**

- Ensure GPU/CUDA compatibility is maintained for changes to the extraction pipeline
- Write clear commit messages describing what changed and why
- Create issues for major changes and get community feedback before submitting PRs
- Be welcoming and respectful to newcomers and contributors from all backgrounds
- Test your changes on at least one PDF and one image poster before submitting

## Your First Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Issues suitable for newcomers
- `help wanted` - Issues where we'd appreciate community help
- `documentation` - Improvements to docs, often good for first-timers

**Never contributed to open source before?** Check out [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github) - it's free!

## Getting Started

### For Code Contributions

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/poster2json.git`
3. Create a branch: `git checkout -b my-feature`
4. Install dependencies: `pip install -e .` or `pip install -r requirements.txt`
5. Make your changes
6. Test with the example posters: `python poster_extraction.py --annotation-dir ./example_posters --output-dir ./test_output`
7. Commit your changes with a clear message
8. Push to your fork and submit a Pull Request

### For Small/Obvious Fixes

Small contributions like fixing typos, improving comments, or correcting documentation can be submitted directly as a PR without creating an issue first.

Examples of obvious fixes:
- Spelling/grammar corrections
- Typo fixes in code comments or documentation
- Formatting improvements
- Updates to metadata files (requirements.txt, .gitignore, etc.)

## How to Report a Bug

### Security Issues

**If you find a security vulnerability, do NOT open a public issue.** Please email the maintainers directly at the contact listed in the repository.

### Bug Reports

When filing a bug report, please include:

1. **What version** are you using? (commit hash or release tag)
2. **What GPU/CUDA version** are you running?
3. **What poster file** caused the issue? (attach if possible, or describe format)
4. **What did you expect** to happen?
5. **What actually happened?** (include error messages/logs)
6. **Steps to reproduce** the issue

## Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to functions
- Keep functions focused and modular

## Questions?

Open an issue with the `question` label and we'll do our best to help!

---

Thank you for contributing to poster2json and helping make scientific poster metadata more FAIR! 🎉