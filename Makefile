
.PHONY: precommit
precommit:
	@poetry run pre-commit run --all


.PHONY: format
format:
	@poetry run isort moments_dnns
	@poetry run black moments_dnns


.PHONY: test
test:
	@poetry run pytest tests


_bump_patch:
	@poetry version patch

_bump_minor:
	@poetry version minor

_bump_major:
	@poetry version major

_update:
	$(eval NEXT_VERSION=$(shell poetry version | sed -e 's/moments-dnns[ ]//g'))
	@git stash
	@git checkout master
	@git pull
	@git checkout -b "chore/release-v${NEXT_VERSION}"
	@git stash apply
	@git-changelog -s conventional -o CHANGELOG.md .
	@git add -u
	@git commit -m "chore: bump to v${NEXT_VERSION}"
	@git push --set-upstream origin "chore/release-v${NEXT_VERSION}"

patch-update: _bump_patch _update

minor-update: _bump_minor _update

major-update: _bump_major _update

