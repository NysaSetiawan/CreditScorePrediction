import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
Modelbucket = "credit-demo-333"
model_s3_key = "model/model.tar.gz"
endpoint_name = "credit-score-endpointv28"
region = "us-east-1"
instance_type = "ml.m5.xlarge"
framework_version = "1.2-1" 

def get_lab_role_arn() -> str:
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]

def main() -> None:
    boto3.setup_default_session(region_name=region)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()
    model_s3_uri = f"s3://{Modelbucket}/{model_s3_key}"

    print(f"Role ARN:  {role_arn}")
    print(f"Model URI: {model_s3_uri}")

    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src",
        framework_version=framework_version,
        py_version="py3",
        sagemaker_session=sm_session,
    )

    print("\nDeploying real-time endpoint...")
    try:
        sm_session.delete_endpoint(endpoint_name)
    except:
        pass

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    sample = {
        "instances": [
            [1,"0x2b4a", "CUST-9999", "Nysa Setiawan", "000-00-0000", "August", 23, 
             "Scientist", 19114.12, 1826.176667, 3, 4, 3, 4, "No Loan", 3, 7, 
             11.27, 4, "Good", 142, 26.272657, "22 Years and 4 Months", "No", 
             312.457143, 60.10134, "Low_spent_Small_value_payments", 186.2667]
        ]
    }

    runtime = boto3.client("sagemaker-runtime", region_name=region)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=str(sample).replace("'", '"'),
    )
    
    print("\nCredit test response:")
    print(response["Body"].read().decode("utf-8"))

if __name__ == "__main__":
    main()