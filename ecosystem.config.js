module.exports = {
  apps: [{
    name: 'multi-agent-api',
    script: 'python',
    args: 'generaic_agent.py',
    cwd: './venv/Scripts',
    interpreter: 'python',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'development',
      PORT: 8000,
      PYTHONPATH: '.',
      // Add your environment variables here
      // ANTHROPIC_API_KEY: 'your_key_here',
      // EMAIL_PASSWORD: 'your_email_password_here'
    },
    env_production: {
      NODE_ENV: 'production',
      PORT: 8000
    },
    log_file: './logs/combined.log',
    out_file: './logs/out.log',
    error_file: './logs/error.log',
    log_date_format: 'YYYY-MM-DD HH:mm Z'
  }]
};
