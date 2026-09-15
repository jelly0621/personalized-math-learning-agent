# Railway Deployment

MathLearningAgent 可以使用 Railway 的 Python/Railpack 构建流程直接部署，不需要 Dockerfile 或 Railway Config-as-Code 文件。

## Deploy

1. 将安全检查后的代码推送到 GitHub repository。
2. 在 Railway 选择 **New Project** → **Deploy from GitHub repo**。
3. 选择本仓库并让 Railway 构建 Python 项目。
4. 设置 Start Command：

   ```text
   python -m scripts.run_web_app
   ```

5. 设置 Health Check Path：

   ```text
   /api/health
   ```

6. 添加 Railway Volume，mount path 设置为：

   ```text
   /app/data
   ```

## Variables

在 Railway Dashboard 中设置以下变量：

```text
LLM_PROVIDER=<provider_name>
LLM_API_KEY=<your_deepseek_api_key>
LLM_BASE_URL=<openai_compatible_base_url>
LLM_MODEL=<text_model_name>
VISION_MODEL=deepseek-v4-flash-vision-exp
DATABASE_PATH=/app/data/math_agent.db
APP_ACCESS_USERNAME=<your_username>
APP_ACCESS_PASSWORD=<strong_password>
```

不要手动设置 `PORT`；Railway 会自动注入。应用检测到 `PORT` 后会监听 `0.0.0.0:$PORT`。`APP_ACCESS_USERNAME` 和 `APP_ACCESS_PASSWORD` 必须同时设置或同时省略；公网部署建议使用强密码并同时设置二者。

## Networking and Verification

在 Railway Networking 中选择 **Generate Domain**，然后依次验证：

- `GET /api/health` 无需认证并返回正常状态。
- 打开 Web 页面时浏览器要求 HTTP Basic Auth。
- 页面与静态资源正常加载。
- 照片上传和多题确认正常。
- 手动或批量错题录入正常。
- Student Profile、Training Plan 和 Online Training 正常。

验证过程会调用已配置的付费模型 API，请使用少量测试数据。

## Persistence Check

录入一条不含个人隐私的测试数据，确认 Profile 或 History 中可见；随后在 Railway 执行 restart 或 redeploy，再次检查该记录仍然存在。若记录保留，说明 `/app/data` Volume 与 `DATABASE_PATH=/app/data/math_agent.db` 配置正确。
