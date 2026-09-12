# Publish with GitHub Desktop and Zenodo

These steps use graphical interfaces only. The prepared package is a local release candidate; nothing has been uploaded to GitHub or published on Zenodo automatically.

## GitHub

1. Download and extract `physics-guided-learning-github-v1.0.0.zip` using Windows Explorer.
2. Open GitHub Desktop and sign in. Choose **File → New repository**. Name it **physics-guided-learning**. Suggested description: **Controlled SciML experiments on perturbation structure, imperfect physics, and augmentation selection in neural PDE forecasting.** Select your normal local repository parent folder. Leave generated README, Git ignore, and license options unset because the package contains these files. Click **Create repository**.
3. Choose **Repository → Show in Explorer**. In another Explorer window, open the extracted `physics-guided-learning` folder. Copy its entire contents into the newly created repository folder. `README.md`, `experiments`, and `results` should appear directly at the top level. The ZIP has already been extracted; copy the files, not the ZIP itself. Keep the package's metadata files, including its Git and Zenodo configuration files.
4. Return to GitHub Desktop. Check that `results/development` and `results/confirmation` appear among the changes. Enter **Initial research release** as the commit summary and click **Commit to main**.
5. Click **Publish repository**. Clear **Keep this code private** for a public research release, then click **Publish Repository**. The first upload contains the archived evaluation arrays and may take several minutes.
6. Choose **Repository → View on GitHub** and confirm that the README, source, and both result folders are visible.

If you already created the repository on GitHub's website, use **File → Clone repository** in Desktop instead of creating another one. Then copy the package contents into that clone, commit, and click **Push origin**. Fetching an existing repository does not add a separate repository to Desktop.

Official guide: [Creating your first repository with GitHub Desktop](https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop).

## Zenodo DOI through the GitHub connection

1. Sign into your existing [Zenodo account](https://zenodo.org/). In the profile menu, open **Linked accounts** and connect GitHub if necessary.
2. Open **GitHub** from the Zenodo profile menu. Click **Sync now**, locate **physics-guided-learning**, and turn its switch on. Do this before publishing the release in the next step.
3. Return to the repository on GitHub. Open **Releases** and choose **Draft a new release** or **Create a new release**. Create the tag **v1.0.0** targeting **main**. Use **Physics-guided learning: code and results v1.0.0** as the release title. Copy the release notes from `docs/RELEASE_NOTES.md`. Publish the release as a normal release.
4. Return to Zenodo's GitHub page, open the repository entry, and wait for the archive to finish. Open the DOI link. Check the author, affiliation, title, license, version, and downloadable archive. Confirm that both results folders are inside that archive.
5. Save the **version DOI** for citing the exact release in the manuscript. The **concept DOI** identifies the project across releases and can be used for its general repository badge.
6. On GitHub, open `README.md` and click the pencil icon to add the DOI link or Zenodo-provided badge. You can also add the real DOI to `CITATION.cff`. Commit the edits to `main`. This changes the working branch; the existing `v1.0.0` release continues to identify the original frozen files. Publish a new version if you later change the scientific code or results.

The first package contains no DOI because the GitHub integration assigns it after release publication. Do not make a separate manual Zenodo upload of the identical package; the integration creates the record. The included `.zenodo.json` supplies Zenodo metadata and takes precedence over `CITATION.cff`; keep author, title, version, and license consistent when editing them. It intentionally contains no DOI.

All evidence is committed inside the repository. The procedure does not rely on separate GitHub release attachments being included in Zenodo's archive.

Official references, checked 2026-09-12:

- [Link an account](https://help.zenodo.org/docs/profile/linking-accounts/)
- [Enable a GitHub repository](https://help.zenodo.org/docs/github/enable-repository/)
- [Create a GitHub release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [Archive a release on Zenodo](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [Describe the software and set metadata](https://help.zenodo.org/docs/github/describe-software/)
- [DOI versioning](https://zenodo.org/help/versioning)
- [GitHub integration cannot reserve a DOI before release](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release)
