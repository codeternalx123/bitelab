import emails
from typing import Dict, Any
from app.core.config import settings
from pathlib import Path
import logging

def send_email(
    email_to: str,
    subject_template: str,
    html_template: str,
    environment: Dict[str, Any]
) -> None:
    try:
        message = emails.Message(
            subject=subject_template,
            html=html_template,
            mail_from=(settings.EMAILS_FROM_NAME, settings.EMAILS_FROM_EMAIL)
        )
        
        response = message.send(
            to=email_to,
            smtp={
                "host": settings.SMTP_HOST,
                "port": settings.SMTP_PORT,
                "tls": settings.SMTP_TLS,
                "user": settings.SMTP_USER,
                "password": settings.SMTP_PASSWORD,
            }
        )
        
        logging.info(f"Email sent to {email_to}. Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Failed to send email to {email_to}. Error: {str(e)}")

def send_reset_password_email(email_to: str, token: str) -> None:
    project_name = settings.APP_NAME
    subject = f"{project_name} - Password recovery"
    
    html_template = f"""
        <p>Password Reset Request</p>
        <p>
            To reset your password, click on the following link:
            <a href="http://localhost:3000/reset-password?token={token}">
                Reset Password
            </a>
        </p>
        <p>
            If you didn't request a password reset, please ignore this email.
            The link is valid for {settings.PASSWORD_RESET_TOKEN_EXPIRE_HOURS} hours.
        </p>
    """
    
    send_email(
        email_to=email_to,
        subject_template=subject,
        html_template=html_template,
        environment={
            "project_name": settings.APP_NAME,
            "token": token,
        }
    )