# how to use mkdocs

Zach is new to mkdocs, but it's pretty easy if you follow christinahedges's very useful tutorial [here](https://christinahedges.github.io/astronomy_workflow/notebooks/3.0-building/mkdocs.html), here's what we're doing:

We start by running `pip install '.[develop]'` in the repository directory, which will naturally install all of the mkdocs requirements.

Then, we set up the docs by going into the base directory for this repository, and running
`mkdocs new .`
which made a `docs/` directory and a `mkdocs.yml`

Then, we edit the `docs/api.md` file to point to the objects and methods we want to explain.

Then, we use the `mkdocs-jupyter` plugin to be able use jupyter notebooks as the source for writing docs, following the examples on their pages. We add some notebooks to the `docs/` diretory and point to them in the `mkdocs.yml` file.

Then, we run `mkdocs serve`, and woah, a live version of the docs appears at http://127.0.0.1:8000/. It's particularly cool (and way better than `sphinx`) that we can make a change to any of the files and simply reload the page to see them update live into the docs. Hooray!

Then, we run `mkdocs gh-deploy`, and double woah, it deployed a pretty version of the docs up at zkbt.github.io/[package-name]!
