# holographic-forecaster-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: holographic-forecaster
spec:
  replicas: 3
  selector:
    matchLabels:
      app: holographic-forecaster
  template:
    metadata:
      labels:
        app: holographic-forecaster
    spec:
      containers:
      - name: forecaster-api
        image: retail-forecaster:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/models/holographic_forecaster.h5"
        volumeMounts:
        - name: model-volume
          mountPath: /models
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: forecaster-service
spec:
  selector:
    app: holographic-forecaster
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
