# DoppelBot: Replace your CEO with an LLM

<p align="center">
  <a href="https://aksh-at--doppel.modal.run/slack/install"><img alt="Add to Slack" height="40" width="139" src="https://platform.slack-edge.com/img/add_to_slack.png" srcSet="https://platform.slack-edge.com/img/add_to_slack.png 1x, https://platform.slack-edge.com/img/add_to_slack@2x.png 2x"  target="_blank" rel="noopener noreferrer" /></a>
</p>

DoppelBot is a Slack app that scrapes a target user's messages in Slack and fine-tunes a large language model (OpenLLaMa) to learn how to respond like them.

<p align="center">

  <img width="489" alt="doppel-bot in action" src="https://github.com/modal-labs/doppel-bot/assets/5786378/4075e372-3a84-4dd3-9ed2-8beaeb18e0d2">
</p>

All the components, including fine-tuning, inference and scraping are serverless and run on [Modal](https://modal.com).

## How it works

[Read the blog post](https://modal.com/docs/guide/slack-finetune).

## Usage

- [Install](https://aksh-at--doppel.modal.run/slack/install) the app to your Slack workspace.
- In any channel, run `/doppel <user>`. Here, `<user>` is either the slack handle or real name of the user you want to target. _Note: for now, we limit each workspace to one target user, and this cannot be changed after installation._
- Wait for the bot to finish training (typically an hour). You can run the command above again to check the status.

<p align="center">
  <img width="489" alt="/doppel command" src="https://github.com/modal-labs/doppel-bot/assets/5786378/9bca0534-5898-4a02-968b-93095ac52b66">
</p>

- Optional: rename the bot to `<user>-bot` (or whatever you want).
  - Go to the [Manage Apps](https://app.slack.com/apps-manage/) page and find `DoppelBot`.
  - Click on `App Details`.
  - Click on `Configuration`.
  - Scroll down to the section named `Bot User`. Click on `Edit` to change the name.

<p align="center">
  <img width="489" alt="/doppel command" src="https://github.com/modal-labs/doppel-bot/assets/5786378/c11c5e24-94ed-4fa0-a445-fa7c6010dc10">
</p>

- In any public Slack channel, including `@doppel` (or the name above if you changed it) in a message will summon the bot.

## Development

This repo contains everything you need to run DoppelBot for yourself.

### Set up Modal

- Create a [Modal](http://modal.com/) account. Note that we have a waitlist at the moment—[reach out](mailto:akshat@modal.com) if you would like to be taken off it sooner.
- Install `modal-client` in your current Python virtual environment (`pip install modal-client`).
- Set up a Modal token in your environment (`modal token new`).

### Create a Slack app

- Go to [https://api.slack.com/apps](https://api.slack.com/apps) and click
  **Create New App**.
- Select **From scratch** if asked _how_ you want to create your app.
- Name your app and select your workspace.
- Go to **Features** > **OAuth & Permissions** on the left navigation pane.
  Under the **Scopes** > **Bot Token Scopes** section, add the following scopes:
  - `app_mentions:read`
  - `channels:history`
  - `channels:join`
  - `channels:read`
  - `chat:write`
  - `chat:write.customize`
  - `commands`
  - `users.profile:read`
  - `users:read`
- On the same page, under the **OAuth tokens for Your Workspace** section,
  click **Install to Workspace** (or reinstall if it's already installed).
- Create a Modal secret
  - On the [create secret page](https://modal.com/secrets/create), select **Slack** as the type.
  - Back on the Slack app settings page, go to **Settings** > **Basic Information** on the left navigation pane.
    Under **App Credentials**, copy the **Signing Secret** and paste its value with the key `SLACK_SIGNING_SECRET`.
  - Go to **OAuth & Permissions** again and copy the **Bot User OAuth Token** and
    paste its value with the key `SLACK_BOT_TOKEN`.
  - Name this secret `slack-finetune-secret`.

### (Optional) Set up Weights & Biases

To track your fine-tuning runs on [Weights & Biases](https://wandb.ai), you'll need to create a Weights & Biases account, and then [create a Modal secret](https://modal.com/secrets/create) with the credentials (click on **Weights & Biases** in the secrets wizard and follow the steps). Then, set [`WANDB_PROJECT`](https://github.com/modal-labs/doppel-bot/blob/aae3f8675e9052251690997557aa8d4a9ae447e6/src/common.py#L10) in `src/common.py` to the name of the project you want to use.

### Deploy your app

From the root directory of this repo, run `modal deploy src.bot`. This will deploy the app to Modal, and print a URL to the terminal (something like `https://aksh-at--doppel.modal.run/`).

Now, we need to point our Slack app to this URL:

- Go to **Features** > **Event Subscriptions** on the left navigation pane:
  - Turn it on.
  - Paste the URL from above into the **Request URL** field, and wait for it to be verified.
  - Under **Subscribe to bot events**, click on **Add bot user event** and add `@app_mention`.
  - Click **Save Changes**.
- Go to **Features** > **Slash Commands** on the left navigation pane. Click **Create New Command**. Set the command to `/doppel` and the request URL to the same URL as above.
- Return to the **Basic Information** page, and click **Install to Workspace**.

### (Optional) Multi-workspace app

If you just want to run the app in your own workspace, the above is all you need. If you want to distribute the app to others, you'll need to set up a multi-workspace app. To enable this, set [`MULTI_WORKSPACE_APP`](https://github.com/modal-labs/doppel-bot/blob/aae3f8675e9052251690997557aa8d4a9ae447e6/src/common.py#L8) to `True` in `src/common.py`.

Then, you'll need to set up [Neon](https://neon.tech/), a serverless Postgres database, for storing user data:

- Create an account and a database on [Neon](https://neon.tech/).
- Create a Modal secret with DB credentials.
  - On the [create secret page](https://modal.com/secrets/create), select **Postgres** as the type.
  - Fill out the values based on the host URL, database name, username and password from Neon. [This page](https://neon.tech/docs/connect/connect-from-any-app) has an example for what it should look like.
  - Name this secret `neon-secret`.
- Create tables by running `modal run src.db` from the root directory of this repo.
- On the Slack app settings page, go to **Settings** > **Manage Distribution**. The **Redirect URLs** should be be `https://<your-modal-run-url>/slack/oauth_redirect`, where `<your-modal-run-url>` is the URL you received after deploying the app above. Once everything looks good, click **Activate Public Distribution**.

Now, deploying the app with `modal deploy src.bot` will take care of setting up all the [intricacies of OAuth](https://api.slack.com/authentication/oauth-v2) for you, and create a multi-workspace Slack app that can be installed by anyone. By default, the install link is at `https://<your-modal-run-url>/slack/install`.

### (Optional) Running each step manually

If you wish, you can also run each step manually. This is useful for debugging or iterating on a specific function.

- Scraper: `modal run src.scrape::scrape --user="<user>"`
- Fine-tuning: `modal run --detach src.finetune --user="<user"` (note that the `--detach` lets you ctrl+c any time without killing the training run)
- Inference: `modal run src.inference --user="<user>"`
