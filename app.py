import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import joblib
import os
import io
import base64
import traceback

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🔒",
    layout="wide"
)

COLUMNS_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
]

FEATURE_DESCRIPTIONS = {
    "duration": "Length of connection in seconds",
    "protocol_type": "Type of protocol (e.g., tcp, udp, icmp)",
    "service": "Network service on destination (e.g., http, ftp)",
    "flag": "Normal or error status of connection (e.g., SF, REJ)",
    "src_bytes": "Bytes sent from source to destination",
    "dst_bytes": "Bytes sent from destination to source",
    "land": "1 if connection is from/to same host/port; 0 otherwise",
    "wrong_fragment": "Number of wrong fragments",
    "urgent": "Number of urgent packets",
    "hot": "Number of 'hot' indicators (e.g., login attempts)",
    "num_failed_logins": "Number of failed login attempts",
    "logged_in": "1 if successfully logged in; 0 otherwise",
    "num_compromised": "Number of compromised conditions",
    "root_shell": "1 if root shell is obtained; 0 otherwise",
    "su_attempted": "1 if 'su root' command attempted; 0 otherwise",
    "num_root": "Number of 'root' accesses",
    "num_file_creations": "Number of file creation operations",
    "num_shells": "Number of shell prompts",
    "num_access_files": "Number of operations on access control files",
    "num_outbound_cmds": "Number of outbound commands in an ftp session",
    "is_host_login": "1 if the login belongs to the 'hot' list; 0 otherwise",
    "is_guest_login": "1 if the login is a 'guest' login; 0 otherwise",
    "count": "Number of connections to the same host in the past 2 seconds",
    "srv_count": "Number of connections to the same service in the past 2 seconds",
    "serror_rate": "% of connections that have 'SYN' errors",
    "srv_serror_rate": "% of connections to the same service that have 'SYN' errors",
    "rerror_rate": "% of connections that have 'REJ' errors",
    "srv_rerror_rate": "% of connections to the same service that have 'REJ' errors",
    "same_srv_rate": "% of connections to the same service",
    "diff_srv_rate": "% of connections to different services",
    "srv_diff_host_rate": "% of connections to different hosts",
    "dst_host_count": "Number of connections to the same destination host",
    "dst_host_srv_count": "Number of connections to the same service from the same destination host",
    "dst_host_same_srv_rate": "% of connections to the same service from the destination host",
    "dst_host_diff_srv_rate": "% of connections to different services from the destination host",
    "dst_host_same_src_port_rate": "% of connections from the same source port to the destination host",
    "dst_host_srv_diff_host_rate": "% of connections to the same service coming from different hosts",
    "dst_host_serror_rate": "% of connections to the destination host that have 'SYN' errors",
    "dst_host_srv_serror_rate": "% of connections to the same service on destination host that have 'SYN' errors",
    "dst_host_rerror_rate": "% of connections to the destination host that have 'REJ' errors",
    "dst_host_srv_rerror_rate": "% of connections to the same service on destination host that have 'REJ' errors"
}

CAT_COLS = ['protocol_type', 'service', 'flag']

@st.cache_data
def load_and_preprocess_data(file):
    try:
        if isinstance(file, str):
            data = pd.read_csv(file, header=None, names=COLUMNS_NAMES)
        else:
            data = pd.read_csv(file, header=None, names=COLUMNS_NAMES)

        st.info(f"Original data shape: {data.shape}")

        if 'last_flag' in data.columns:
            data = data.drop('last_flag', axis=1)
            st.info("Dropped 'last_flag' column.")

        if data.empty:
            st.error("Uploaded data is empty.")
            return None, None, None, None

        if 'attack' in data.columns:
            data['attack'] = data['attack'].astype(str).str.rstrip('.')
        else:
            st.error("The crucial 'attack' column is missing from the data.")
            return None, None, None, None

        encoders = {}
        for col in CAT_COLS:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = data[col].astype(str).fillna('missing')
                data[col] = le.fit_transform(data[col])
                encoders[col] = le
            else:
                st.warning(f"Categorical column '{col}' not found in the uploaded data.")

        data['attack_flag'] = data['attack'].apply(lambda x: 0 if x == 'normal' else 1)

        attack_types = data['attack'].unique()
        attack_mapping = {attack: idx for idx, attack in enumerate(attack_types)}
        reverse_mapping = {idx: attack for attack, idx in attack_mapping.items()}
        data['attack_type'] = data['attack'].map(attack_mapping)

        st.success(f"Preprocessing complete. Final data shape: {data.shape}")
        return data, attack_mapping, reverse_mapping, encoders

    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}")
        st.error("Please ensure the file is a CSV/TXT with the correct KDD'99 structure (41 features + 1 label).")
        st.error(traceback.format_exc())
        return None, None, None, None

@st.cache_resource
def train_model(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def get_download_link(obj, filename, link_text):
    try:
        if isinstance(obj, pd.DataFrame):
            data = obj.to_csv(index=False).encode()
            mime_type = "text/csv"
        else:
            buffer = io.BytesIO()
            joblib.dump(obj, buffer)
            buffer.seek(0)
            data = buffer.read()
            mime_type = "application/octet-stream"

        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link for {filename}: {e}")
        return "Download link generation failed."

def create_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame()

    data['duration'] = np.random.exponential(scale=1.0, size=n_samples).round(1)
    data['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'], size=n_samples, p=[0.7, 0.2, 0.1])
    data['service'] = np.random.choice(['http', 'private', 'smtp', 'ftp_data', 'other', 'ecr_i', 'domain_u'], size=n_samples, p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.08, 0.07])
    data['flag'] = np.random.choice(['SF', 'REJ', 'S0', 'S1', 'RSTR'], size=n_samples, p=[0.6, 0.15, 0.15, 0.05, 0.05])
    data['src_bytes'] = (np.random.exponential(scale=500, size=n_samples) * (1 + (data['protocol_type'] == 'tcp') * 2)).astype(int)
    data['dst_bytes'] = (np.random.exponential(scale=300, size=n_samples) * (1 + (data['protocol_type'] == 'tcp') * 5)).astype(int)
    data['land'] = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
    data['wrong_fragment'] = np.random.choice([0, 1, 2], size=n_samples, p=[0.95, 0.04, 0.01])
    data['urgent'] = np.random.choice([0, 1], size=n_samples, p=[0.999, 0.001])

    data['hot'] = np.random.poisson(lam=0.1, size=n_samples)
    data['num_failed_logins'] = np.random.poisson(lam=0.01, size=n_samples)
    data['logged_in'] = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    data['num_compromised'] = np.random.poisson(lam=0.05, size=n_samples)
    data['root_shell'] = np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])
    data['su_attempted'] = np.random.choice([0, 1], size=n_samples, p=[0.998, 0.002])
    data['num_root'] = np.random.poisson(lam=0.02, size=n_samples)
    data['num_file_creations'] = np.random.poisson(lam=0.03, size=n_samples)
    data['num_shells'] = np.random.choice([0, 1], size=n_samples, p=[0.997, 0.003])
    data['num_access_files'] = np.random.poisson(lam=0.01, size=n_samples)
    data['num_outbound_cmds'] = 0
    data['is_host_login'] = np.random.choice([0, 1], size=n_samples, p=[0.999, 0.001])
    data['is_guest_login'] = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

    data['count'] = np.random.randint(1, 150, size=n_samples)
    data['srv_count'] = (data['count'] * np.random.uniform(0.5, 1.0)).astype(int)
    data['serror_rate'] = np.random.beta(a=0.5, b=5, size=n_samples).round(2)
    data['srv_serror_rate'] = (data['serror_rate'] * np.random.uniform(0.8, 1.0)).clip(0, 1).round(2)
    data['rerror_rate'] = np.random.beta(a=0.3, b=6, size=n_samples).round(2)
    data['srv_rerror_rate'] = (data['rerror_rate'] * np.random.uniform(0.8, 1.0)).clip(0, 1).round(2)
    data['same_srv_rate'] = (1 - data['rerror_rate'] - data['serror_rate'] * np.random.uniform(0.8, 1.0)).clip(0, 1).round(2)
    data['diff_srv_rate'] = (1 - data['same_srv_rate'] * np.random.uniform(0.9, 1.0)).clip(0, 1).round(2)
    data['srv_diff_host_rate'] = np.random.beta(a=0.2, b=5, size=n_samples).round(2)

    data['dst_host_count'] = np.random.randint(1, 256, size=n_samples)
    data['dst_host_srv_count'] = (data['dst_host_count'] * data['same_srv_rate'] * np.random.uniform(0.7, 1.0)).clip(1, 255).astype(int)
    data['dst_host_same_srv_rate'] = data['same_srv_rate']
    data['dst_host_diff_srv_rate'] = data['diff_srv_rate']
    data['dst_host_same_src_port_rate'] = np.random.beta(a=0.5, b=4, size=n_samples).round(2)
    data['dst_host_srv_diff_host_rate'] = data['srv_diff_host_rate']
    data['dst_host_serror_rate'] = data['serror_rate']
    data['dst_host_srv_serror_rate'] = data['srv_serror_rate']
    data['dst_host_rerror_rate'] = data['rerror_rate']
    data['dst_host_srv_rerror_rate'] = data['srv_rerror_rate']

    attacks = ['normal'] * int(n_samples * 0.8) + \
              ['neptune'] * int(n_samples * 0.05) + \
              ['smurf'] * int(n_samples * 0.04) + \
              ['portsweep'] * int(n_samples * 0.03) + \
              ['satan'] * int(n_samples * 0.03) + \
              ['ipsweep'] * int(n_samples * 0.03) + \
              ['back'] * int(n_samples * 0.02)
    if len(attacks) < n_samples:
        attacks.extend(['normal'] * (n_samples - len(attacks)))
    attacks = attacks[:n_samples]
    np.random.shuffle(attacks)
    data['attack'] = attacks
    data['last_flag'] = np.random.randint(0, 20, size=n_samples)

    final_columns = [col for col in COLUMNS_NAMES if col in data.columns]
    data = data[final_columns]

    return data

def main():
    st.title("🔒 Network Intrusion Detection System")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Overview", "Data Exploration", "Model Training", "Real-time Prediction", "Batch Prediction", "Help"])

    if 'data' not in st.session_state:
        st.session_state.data = None
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.encoders = None
        st.session_state.attack_mapping = None
        st.session_state.reverse_mapping = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.selected_features = None

    with st.sidebar.expander("📁 Upload & Load Data", expanded=True):
        uploaded_file = st.file_uploader("Upload Network Traffic Data (CSV/TXT - KDD'99 Format)", type=['csv', 'txt'])
        use_sample_data = st.checkbox("Use built-in sample data", value=(st.session_state.data is None and uploaded_file is None))

        if st.button("🔄 Load Data"):
            with st.spinner('Loading and preprocessing data...'):
                data_source = None
                if use_sample_data:
                    st.info("Generating and loading sample data...")
                    synthetic_data = create_synthetic_data(n_samples=5000)
                    csv_data = synthetic_data.to_csv(index=False, header=False)
                    data_source = io.StringIO(csv_data)
                    st.sidebar.success("Sample data generated.")
                elif uploaded_file is not None:
                    data_source = uploaded_file
                    st.sidebar.success("File uploaded.")
                else:
                    st.sidebar.warning("Please upload a file or check the sample data box.")

                if data_source:
                    data, attack_mapping, reverse_mapping, encoders = load_and_preprocess_data(data_source)
                    if data is not None:
                        st.session_state.data = data
                        st.session_state.attack_mapping = attack_mapping
                        st.session_state.reverse_mapping = reverse_mapping
                        st.session_state.encoders = encoders
                        st.session_state.model = None
                        st.session_state.scaler = None
                        st.session_state.X_train = None
                        st.session_state.X_test = None
                        st.session_state.y_train = None
                        st.session_state.y_test = None
                        st.session_state.selected_features = None
                        st.sidebar.success(f"Data loaded! Shape: {data.shape}")
                    else:
                        st.sidebar.error("Data loading failed. Check file format and content.")
                else:
                    st.sidebar.warning("No data source selected.")

    if page == "Overview":
        st.header("Welcome to the Network Intrusion Detection System")
        st.markdown("""
        This application leverages machine learning to detect potential intrusions and malicious activities within network traffic data. Analyze patterns, train detection models, and classify connections in real-time or batch mode.

        **Key Features:**
        *   **📊 Data Exploration:** Visualize traffic distributions, correlations, and feature characteristics.
        *   **🧠 Model Training:** Train a Random Forest classifier for binary (Normal/Attack) or multiclass detection. Evaluate performance and feature importance.
        *   **⚡ Real-time Prediction:** Input individual connection details to classify them instantly.
        *   **📦 Batch Prediction:** Upload a file containing multiple connections for bulk analysis.
        *   **❓ Help & Documentation:** Understand data format, features, and application usage.

        **Getting Started:**
        1.  Use the sidebar to **upload** your network traffic data (CSV/TXT following KDD'99 format, no header) or use the **sample data**.
        2.  Click **Load Data**.
        3.  Navigate through the pages using the sidebar radio buttons.

        ---
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            #### Required Data Format
            *   CSV or TXT file.
            *   No header row.
            *   41 features followed by the attack label (e.g., 'normal', 'neptune'). **Note:** Ensure labels don't have trailing dots.
            *   Based on the [KDD Cup 1999 Dataset structure](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).
            """)
        with col2:
            st.success("""
            #### Application Workflow
            1.  **Load Data:** Upload or use sample data.
            2.  **Explore (Optional):** Understand data patterns.
            3.  **Train Model:** Configure and train the classifier.
            4.  **Predict:** Analyze new connections (real-time or batch).
            """)

        st.warning("""
        **Disclaimer:** This tool is for educational and illustrative purposes. While effective on benchmark datasets, performance on real-world, live network traffic requires careful validation, tuning, and integration into a broader security strategy. It is not a replacement for comprehensive security solutions.
        """)

    elif page == "Data Exploration":
        st.header("📊 Data Exploration")

        if st.session_state.data is None:
            st.warning("⚠️ Please load data first using the sidebar.")
        else:
            data = st.session_state.data

            st.subheader("Dataset Overview")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Number of Connections", data.shape[0])
                non_feature_cols = ['attack', 'attack_flag', 'attack_type']
                num_features = len([col for col in data.columns if col not in non_feature_cols])
                st.metric("Number of Features", num_features)

                if st.checkbox("Show Feature Names"):
                    st.write([col for col in data.columns if col not in non_feature_cols])
            with col2:
                st.dataframe(data.head(), height=210)

            st.subheader("Attack Distribution")
            col1, col2 = st.columns([2,1])
            with col1:
                fig_att, ax_att = plt.subplots(figsize=(10, 6))
                attack_counts = data['attack'].value_counts()
                display_limit = 20
                if len(attack_counts) > display_limit:
                    top_attacks = attack_counts.nlargest(display_limit)
                    other_count = attack_counts.iloc[display_limit:].sum()
                    if other_count > 0:
                        plot_counts = top_attacks.copy()
                        plot_counts['Other'] = other_count
                    else:
                        plot_counts = top_attacks
                    attack_counts_display = plot_counts
                    plot_title = f'Distribution of Top {display_limit} Attack Types (and Other)'
                else:
                    attack_counts_display = attack_counts
                    plot_title = 'Distribution of All Attack Types'

                sns.barplot(x=attack_counts_display.index, y=attack_counts_display.values, ax=ax_att, palette="viridis")
                ax_att.set_title(plot_title)
                ax_att.set_ylabel("Number of Connections")
                ax_att.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig_att)
            with col2:
                st.write("Attack Counts:")
                st.dataframe(attack_counts.head(50))

            st.subheader("Protocol, Service, and Flag Distributions")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_proto, ax_proto = plt.subplots()
                if 'protocol_type' in data.columns and st.session_state.encoders and 'protocol_type' in st.session_state.encoders:
                    encoder = st.session_state.encoders['protocol_type']
                    decoded_protocols = data['protocol_type'].apply(lambda x: encoder.inverse_transform([x])[0] if x in encoder.transform(encoder.classes_) else f"Unknown({x})")
                    sns.countplot(y=decoded_protocols, ax=ax_proto, order=decoded_protocols.value_counts().index, palette="Blues")
                    ax_proto.set_title('Protocol Type Distribution')
                    ax_proto.set_ylabel("Protocol")
                    ax_proto.set_xlabel("Count")
                else:
                    ax_proto.text(0.5, 0.5, 'Protocol data or encoder missing', ha='center', va='center')
                st.pyplot(fig_proto)
            with col2:
                fig_serv, ax_serv = plt.subplots()
                if 'service' in data.columns and st.session_state.encoders and 'service' in st.session_state.encoders:
                    encoder = st.session_state.encoders['service']
                    decoded_services = data['service'].apply(lambda x: encoder.inverse_transform([x])[0] if x in encoder.transform(encoder.classes_) else f"Unknown({x})")
                    top_n_services = 15
                    service_counts = decoded_services.value_counts()
                    top_services = service_counts.nlargest(top_n_services).index
                    plot_data_serv = decoded_services[decoded_services.isin(top_services)]
                    sns.countplot(y=plot_data_serv, order=top_services, ax=ax_serv, palette="Greens")
                    ax_serv.set_title(f'Top {top_n_services} Services')
                    ax_serv.set_xlabel("Count")
                    ax_serv.set_ylabel("Service")
                else:
                    ax_serv.text(0.5, 0.5, 'Service data or encoder missing', ha='center', va='center')
                st.pyplot(fig_serv)
            with col3:
                fig_flag, ax_flag = plt.subplots()
                if 'flag' in data.columns and st.session_state.encoders and 'flag' in st.session_state.encoders:
                    encoder = st.session_state.encoders['flag']
                    decoded_flags = data['flag'].apply(lambda x: encoder.inverse_transform([x])[0] if x in encoder.transform(encoder.classes_) else f"Unknown({x})")
                    top_n_flags = 10
                    flag_counts = decoded_flags.value_counts()
                    top_flags = flag_counts.nlargest(top_n_flags).index
                    plot_data_flag = decoded_flags[decoded_flags.isin(top_flags)]
                    sns.countplot(x=plot_data_flag, order=top_flags, ax=ax_flag, palette="Oranges")
                    ax_flag.set_title(f'Top {top_n_flags} Flags')
                    ax_flag.set_xlabel("Flag")
                    ax_flag.set_ylabel("Count")
                else:
                    ax_flag.text(0.5, 0.5, 'Flag data or encoder missing', ha='center', va='center')
                st.pyplot(fig_flag)

            st.subheader("Feature Correlation Analysis")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            features_for_corr = [col for col in numeric_cols if col not in ['attack_flag', 'attack_type']]
            if features_for_corr:
                default_selection = features_for_corr[:15] if len(features_for_corr) >= 15 else features_for_corr

                selected_features_corr = st.multiselect(
                    "Select features for correlation heatmap:",
                    features_for_corr,
                    default=default_selection
                )

                if selected_features_corr:
                    sample_size = min(len(data), 5000)
                    corr_data = data.sample(sample_size, random_state=42)[selected_features_corr]
                    correlation = corr_data.corr()

                    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
                    mask = np.triu(np.ones_like(correlation, dtype=bool))
                    sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', fmt='.2f', square=True, linewidths=.5, ax=ax_corr)
                    ax_corr.set_title('Feature Correlation Heatmap (Sampled Data)')
                    plt.tight_layout()
                    st.pyplot(fig_corr)

                    if st.checkbox("Show Strongest Correlations (Top 20)", value=False):
                        corr_matrix_unstack = correlation.abs().unstack().sort_values(ascending=False)
                        corr_matrix_unstack = corr_matrix_unstack[corr_matrix_unstack < 1.0]
                        highest_corr_df = pd.DataFrame(corr_matrix_unstack).reset_index()
                        highest_corr_df.columns = ['Feature 1', 'Feature 2', 'Absolute Correlation']
                        highest_corr_df = highest_corr_df.loc[highest_corr_df['Feature 1'] < highest_corr_df['Feature 2']]
                        st.dataframe(highest_corr_df.head(20))
                else:
                    st.info("Select one or more features to display the correlation heatmap.")
            else:
                st.info("No numerical features available for correlation analysis after excluding target variables.")

            st.subheader("Feature Distribution Analysis")
            if features_for_corr:
                default_feature_index = 0
                if 'duration' in features_for_corr:
                    default_feature_index = features_for_corr.index('duration')

                feature_to_visualize = st.selectbox(
                    "Select a numerical feature to visualize its distribution:",
                    features_for_corr,
                    index=default_feature_index,
                )

                if feature_to_visualize:
                    st.info(f"**Feature description:** {FEATURE_DESCRIPTIONS.get(feature_to_visualize, 'No description available.')}")
                    log_scale = st.checkbox(f"Use Log Scale for '{feature_to_visualize}' (useful for skewed data)")

                    fig_dist, axes = plt.subplots(1, 2, figsize=(15, 5))

                    plot_data_hist = data[[feature_to_visualize, 'attack_flag']].copy()
                    feature_log_applied = False
                    if log_scale:
                        if (plot_data_hist[feature_to_visualize] <= 0).any():
                            st.warning(f"Feature '{feature_to_visualize}' contains non-positive values. Adding 1 before applying log scale for visualization.")
                            plot_data_hist[feature_to_visualize] = plot_data_hist[feature_to_visualize] + 1
                        if (plot_data_hist[feature_to_visualize] > 0).all():
                            plot_data_hist[feature_to_visualize] = np.log(plot_data_hist[feature_to_visualize])
                            hist_title = f'Log Distribution of {feature_to_visualize}'
                            feature_log_applied = True
                        else:
                            st.error(f"Cannot apply log scale to '{feature_to_visualize}' even after adding 1.")
                            hist_title = f'Distribution of {feature_to_visualize} (Log scale failed)'
                    else:
                        hist_title = f'Distribution of {feature_to_visualize}'

                    sns.histplot(data=plot_data_hist, x=feature_to_visualize, hue='attack_flag', kde=True, element="step", common_norm=False, ax=axes[0], palette={0: "blue", 1: "red"})
                    axes[0].set_title(hist_title)
                    handles, _ = axes[0].get_legend_handles_labels()
                    axes[0].legend(handles, ['Normal (0)', 'Attack (1)'], title='Type')

                    plot_data_box = plot_data_hist
                    box_title = f'Box Plot of {"Log " if feature_log_applied else ""}{feature_to_visualize} by Attack Flag'

                    sns.boxplot(x='attack_flag', y=feature_to_visualize, data=plot_data_box, ax=axes[1], palette={0: "blue", 1: "red"})
                    axes[1].set_title(box_title)
                    axes[1].set_xticklabels(['Normal (0)', 'Attack (1)'])
                    axes[1].set_xlabel('Attack Flag')

                    plt.tight_layout()
                    st.pyplot(fig_dist)
            else:
                st.info("No numerical features available for distribution analysis.")

    elif page == "Model Training":
        st.header("🧠 Model Training")

        if st.session_state.data is None:
            st.warning("⚠️ Please load data first using the sidebar.")
        else:
            data = st.session_state.data

            st.subheader("Model Configuration")
            col1, col2 = st.columns(2)

            with col1:
                classification_type = st.radio(
                    "Select classification type:",
                    ["Binary (Normal vs Attack)", "Multiclass (All attack types)"],
                    key="classification_type",
                    help="Binary is generally faster and simpler. Multiclass provides detailed attack identification."
                )
                test_size = st.slider("Test set size (%)", 10, 50, 25, 5, key="test_size") / 100
                n_estimators = st.slider("Number of trees (Random Forest)", 50, 300, 100, 50, key="n_estimators")

            with col2:
                st.write("**Feature Selection:**")
                available_features = [col for col in data.columns if col not in ['attack', 'attack_flag', 'attack_type']]

                all_features = st.checkbox("Use all available features", value=True, key="use_all_features")

                if not all_features:
                    selected_features = st.multiselect(
                        "Select features to use for training:",
                        available_features,
                        default=available_features,
                        key="feature_select"
                    )
                    if not selected_features:
                        st.warning("Please select at least one feature.")
                        selected_features = st.session_state.get('selected_features', [])
                    else:
                        st.session_state.selected_features = selected_features
                else:
                    selected_features = available_features
                    st.session_state.selected_features = selected_features

                st.info(f"{len(selected_features)} features selected for training.")

            if st.button("🚀 Train Model", type="primary"):
                if not selected_features:
                    st.error("Cannot train model without selected features.")
                else:
                    with st.spinner('Training model... This might take a few moments.'):
                        try:
                            X = data[selected_features]
                            y = data['attack_flag'] if classification_type == "Binary (Normal vs Attack)" else data['attack_type']

                            non_numeric_cols = X.select_dtypes(exclude=np.number).columns
                            if not non_numeric_cols.empty:
                                st.error(f"Non-numeric columns found in feature set: {list(non_numeric_cols)}. This shouldn't happen after preprocessing. Please check data loading.")
                                st.stop()

                            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                                st.error("NaN or Infinity values detected in the feature set before scaling. Please check data quality.")
                                st.stop()

                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            st.session_state.scaler = scaler

                            X_train, X_test, y_train, y_test = train_test_split(
                                X_scaled, y, test_size=test_size, random_state=42, stratify=y
                            )

                            st.session_state.X_train, st.session_state.X_test = X_train, X_test
                            st.session_state.y_train, st.session_state.y_test = y_train, y_test

                            start_time = time.time()
                            model = train_model(X_train, y_train, n_estimators=n_estimators)
                            training_time = time.time() - start_time
                            st.session_state.model = model

                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)

                            st.success(f"Model trained successfully in {training_time:.2f} seconds!")

                            st.subheader("📊 Model Evaluation Results")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Accuracy", f"{accuracy:.4f}")
                            col2.metric("Training Samples", len(X_train))
                            col3.metric("Testing Samples", len(X_test))

                            eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Classification Report", "Confusion Matrix", "Feature Importance"])

                            with eval_tab1:
                                st.text("Classification Report:")
                                target_names = None
                                if classification_type == "Binary (Normal vs Attack)":
                                    target_names = ["Normal (0)", "Attack (1)"]
                                elif st.session_state.reverse_mapping:
                                    try:
                                        unique_labels_in_y = sorted(y.unique())
                                        target_names = [st.session_state.reverse_mapping.get(l, f"Unknown ({l})") for l in unique_labels_in_y]
                                    except Exception as e:
                                        st.warning(f"Could not map multiclass labels for report: {e}")
                                        target_names = [f"Class {l}" for l in sorted(y.unique())]

                                report = classification_report(y_test, y_pred, output_dict=True, target_names=target_names, zero_division=0)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.round(3))

                            with eval_tab2:
                                st.text("Confusion Matrix:")
                                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                                unique_cm_labels = sorted(list(set(y_test) | set(y_pred)))

                                cm = confusion_matrix(y_test, y_pred, labels=unique_cm_labels)

                                if classification_type == "Binary (Normal vs Attack)":
                                    cm_display_labels = ["Normal", "Attack"]
                                elif st.session_state.reverse_mapping:
                                    try:
                                        cm_display_labels = [st.session_state.reverse_mapping.get(l, f"Unknown ({l})") for l in unique_cm_labels]
                                    except:
                                        cm_display_labels = [f"Class {l}" for l in unique_cm_labels]
                                else:
                                    cm_display_labels = [f"Class {l}" for l in unique_cm_labels]

                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                            xticklabels=cm_display_labels, yticklabels=cm_display_labels)
                                plt.title('Confusion Matrix')
                                plt.xlabel('Predicted Label')
                                plt.ylabel('True Label')
                                plt.tight_layout()
                                st.pyplot(fig_cm)

                            with eval_tab3:
                                st.text("Feature Importance:")
                                if hasattr(model, 'feature_importances_'):
                                    feature_importance = pd.DataFrame({
                                        'Feature': selected_features,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False)

                                    fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
                                    top_n_features = min(20, len(feature_importance))
                                    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n_features), ax=ax_fi, palette="viridis")
                                    ax_fi.set_title(f'Top {top_n_features} Feature Importances')
                                    plt.tight_layout()
                                    st.pyplot(fig_fi)

                                    with st.expander("Show Full Feature Importance List"):
                                        st.dataframe(feature_importance)
                                else:
                                    st.info("The trained model does not provide feature importance information.")

                            st.markdown("---")
                            st.subheader("💾 Download Model & Scaler")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(get_download_link(model, "intrusion_detection_model.pkl", "Download Trained Model (.pkl)"), unsafe_allow_html=True)
                            with col2:
                                if st.session_state.scaler:
                                    st.markdown(get_download_link(st.session_state.scaler, "scaler.pkl", "Download Scaler (.pkl)"), unsafe_allow_html=True)
                                else:
                                    st.info("Scaler not available for download.")
                        except Exception as e:
                            st.error(f"An error occurred during training: {e}")
                            st.error(traceback.format_exc())

    elif page == "Real-time Prediction":
        st.header("⚡ Real-time Connection Analysis")

        if st.session_state.model is None or st.session_state.scaler is None or st.session_state.selected_features is None:
            st.warning("⚠️ Please train a model first on the 'Model Training' page.")
            st.info("Ensure data is loaded and a model is trained using the desired features.")
        else:
            st.info("Enter connection details below. Features should match those used for training.")
            st.markdown(f"**Model trained with {len(st.session_state.selected_features)} features:** `{', '.join(st.session_state.selected_features)}`")

            required_features = st.session_state.selected_features
            encoders = st.session_state.encoders if st.session_state.encoders else {}
            input_data = {}

            with st.form("realtime_prediction_form"):
                st.write("Enter feature values:")
                cols = st.columns(3)

                col_idx = 0
                for feature in required_features:
                    current_col = cols[col_idx % 3]
                    with current_col:
                        help_text = FEATURE_DESCRIPTIONS.get(feature, "No description available.")

                        if feature in CAT_COLS and feature in encoders:
                            try:
                                encoder_obj = encoders[feature]
                                options = list(encoder_obj.classes_)
                                default_index = 0
                                input_data[feature] = st.selectbox(f"{feature}", options, index=default_index, key=feature, help=help_text)
                            except Exception as e:
                                st.error(f"Error creating dropdown for {feature}: {e}")
                                input_data[feature] = st.text_input(f"{feature} (Enter text value)", key=feature, help=help_text)
                        elif feature in ['land', 'logged_in', 'root_shell', 'su_attempted', 'is_host_login', 'is_guest_login']:
                            input_data[feature] = st.selectbox(f"{feature}", [0, 1], index=0, key=feature, help=help_text)
                        else:
                            min_val = 0.0
                            default_val = 0.0
                            format_str = None

                            if "rate" in feature:
                                max_val = 1.0
                                step = 0.01
                                format_str = "%.2f"
                            elif feature == "duration":
                                max_val = None
                                step = 0.1
                                format_str = "%.1f"
                            elif feature in ['wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']:
                                max_val = None
                                step = 1.0
                                format_str = "%.0f"
                            else:
                                max_val = None
                                step = 100.0
                                default_val = 0.0
                                format_str = "%.0f"

                            input_data[feature] = st.number_input(
                                f"{feature}",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                step=step,
                                key=feature,
                                help=help_text,
                                format=format_str
                            )

                    col_idx += 1

                submitted = st.form_submit_button("Analyze Connection")

            if submitted:
                try:
                    input_list = []
                    unknown_categories_found = False
                    for feature in required_features:
                        value = input_data[feature]
                        if feature in CAT_COLS and feature in encoders:
                            encoder = encoders[feature]
                            if value not in encoder.classes_:
                                st.warning(f"Unknown category '{value}' for feature '{feature}'. Mapping to -1 (check encoding).")
                                input_list.append(-1)
                                unknown_categories_found = True
                            else:
                                input_list.append(encoder.transform([value])[0])
                        else:
                            input_list.append(float(value))

                    input_array = np.array(input_list).reshape(1, -1)

                    if np.isnan(input_array).any() or np.isinf(input_array).any():
                        st.error("Invalid numerical input detected after processing. Please check values.")
                        st.stop()

                    input_scaled = st.session_state.scaler.transform(input_array)

                    prediction = st.session_state.model.predict(input_scaled)
                    prediction_proba = st.session_state.model.predict_proba(input_scaled)

                    st.subheader("📈 Analysis Result")
                    prediction_class = prediction[0]
                    is_binary = len(prediction_proba[0]) == 2

                    if is_binary:
                        result_label = "NORMAL" if prediction_class == 0 else "ATTACK"
                        confidence = prediction_proba[0][prediction_class] * 100
                        if result_label == "NORMAL":
                            st.success(f"✅ Connection Classified as: **{result_label}**")
                        else:
                            st.error(f"🚨 Connection Classified as: **{result_label}**")
                        st.metric("Confidence", f"{confidence:.2f}%")

                        fig_proba, ax_proba = plt.subplots(figsize=(5, 3))
                        sns.barplot(x=["Normal", "Attack"], y=prediction_proba[0], ax=ax_proba, palette=["#66b3ff", "#ff9999"])
                        ax_proba.set_ylabel("Probability")
                        ax_proba.set_title("Prediction Probabilities")
                        ax_proba.set_ylim(0, 1)
                        for i, v in enumerate(prediction_proba[0]):
                            ax_proba.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
                        st.pyplot(fig_proba)

                    else:
                        attack_name = st.session_state.reverse_mapping.get(prediction_class, f"Unknown Class ({prediction_class})")
                        confidence = prediction_proba[0][prediction_class] * 100
                        if attack_name == 'normal':
                            st.success(f"✅ Connection Classified as: **NORMAL**")
                        else:
                            st.error(f"🚨 Connection Classified as: **ATTACK ({attack_name})**")
                        st.metric("Confidence", f"{confidence:.2f}%")

                        top_n_classes = min(5, len(prediction_proba[0]))
                        class_indices = np.argsort(prediction_proba[0])[::-1][:top_n_classes]
                        top_probs = prediction_proba[0][class_indices]
                        top_labels = [st.session_state.reverse_mapping.get(i, f"Unknown ({i})") for i in class_indices]

                        fig_proba, ax_proba = plt.subplots(figsize=(7, 4))
                        sns.barplot(x=top_labels, y=top_probs, ax=ax_proba, palette="viridis")
                        ax_proba.set_ylabel("Probability")
                        ax_proba.set_title(f"Top {top_n_classes} Predicted Class Probabilities")
                        ax_proba.set_ylim(0, 1)
                        ax_proba.tick_params(axis='x', rotation=30)
                        for i, v in enumerate(top_probs):
                            ax_proba.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
                        st.pyplot(fig_proba)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.error(traceback.format_exc())

    elif page == "Batch Prediction":
        st.header("📦 Batch Prediction")

        if st.session_state.model is None or st.session_state.scaler is None or st.session_state.selected_features is None:
            st.warning("⚠️ Please train a model first on the 'Model Training' page.")
            st.info("Ensure data is loaded, a model is trained, and features match training.")
        else:
            st.info("Upload a CSV/TXT file with multiple connections (same format as training data, without header).")

            batch_file = st.file_uploader("Upload data file for batch prediction", type=['csv', 'txt'], key="batch_uploader")

            if batch_file is not None:
                try:
                    try:
                        batch_data_raw = pd.read_csv(batch_file, header=None)
                        num_cols = batch_data_raw.shape[1]
                        if num_cols >= len(COLUMNS_NAMES) -1 :
                            batch_data_raw.columns = COLUMNS_NAMES[:num_cols]
                        else:
                            st.error(f"Uploaded file has only {num_cols} columns. Expected at least 41 features based on KDD'99 format.")
                            st.stop()
                    except Exception as e:
                        st.error(f"Error reading the batch file: {e}")
                        st.error("Please ensure it's a valid CSV/TXT file without a header.")
                        st.stop()

                    st.write("Preview of uploaded data (first 5 rows):")
                    st.dataframe(batch_data_raw.head())

                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        with st.spinner("Processing batch data and predicting..."):
                            batch_data_processed = batch_data_raw.copy()

                            encoders = st.session_state.encoders if st.session_state.encoders else {}
                            unknown_categories_summary = {}
                            for feature, encoder in encoders.items():
                                if feature in batch_data_processed.columns:
                                    unknown_value = -1
                                    batch_data_processed[feature] = batch_data_processed[feature].astype(str)
                                    is_known = batch_data_processed[feature].isin(encoder.classes_)
                                    num_unknown = (~is_known).sum()
                                    if num_unknown > 0:
                                        unknown_categories_summary[feature] = num_unknown
                                    batch_data_processed.loc[is_known, feature] = encoder.transform(batch_data_processed.loc[is_known, feature])
                                    batch_data_processed.loc[~is_known, feature] = unknown_value
                                else:
                                    st.warning(f"Categorical column '{feature}' used in training not found in batch file.")

                            if unknown_categories_summary:
                                st.warning("Unknown categories found and mapped to -1:")
                                st.json(unknown_categories_summary)

                            try:
                                required_features = st.session_state.selected_features
                                X_batch = batch_data_processed[required_features]
                            except KeyError as e:
                                st.error(f"Missing required feature in uploaded file: {e}. Ensure the batch file contains all features used for training.")
                                st.stop()

                            X_batch = X_batch.apply(pd.to_numeric, errors='coerce')

                            if X_batch.isnull().values.any():
                                st.warning("NaN values detected after encoding/coercion. Check data quality or unknown categories. Filling with 0 for prediction.")
                                X_batch = X_batch.fillna(0)

                            non_numeric_cols_batch = X_batch.select_dtypes(exclude=np.number).columns
                            if not non_numeric_cols_batch.empty:
                                st.error(f"Non-numeric columns remain after encoding/coercion: {list(non_numeric_cols_batch)}. Cannot proceed.")
                                st.stop()

                            X_batch_scaled = st.session_state.scaler.transform(X_batch)

                            y_pred_batch = st.session_state.model.predict(X_batch_scaled)
                            y_pred_proba_batch = st.session_state.model.predict_proba(X_batch_scaled)

                            results_df = batch_data_raw.copy()
                            results_df['prediction_code'] = y_pred_batch
                            results_df['confidence'] = np.max(y_pred_proba_batch, axis=1) * 100

                            is_binary = len(y_pred_proba_batch[0]) == 2
                            if is_binary:
                                results_df['prediction_label'] = results_df['prediction_code'].apply(lambda x: 'Attack' if x == 1 else 'Normal')
                            else:
                                results_df['prediction_label'] = results_df['prediction_code'].apply(
                                    lambda x: st.session_state.reverse_mapping.get(x, f"Unknown ({x})")
                                )

                            st.success(f"Processed {len(results_df)} connections!")

                            st.subheader("📊 Prediction Summary")
                            pred_counts = results_df['prediction_label'].value_counts()

                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Total Connections Analyzed", len(results_df))
                                if is_binary:
                                    st.metric("Normal Predictions", pred_counts.get("Normal", 0))
                                    st.metric("Attack Predictions", pred_counts.get("Attack", 0))
                                else:
                                    st.metric("Normal Predictions", pred_counts.get("normal", 0))
                                    st.metric("Total Attack Predictions", sum(count for label, count in pred_counts.items() if label != 'normal'))

                            with col2:
                                fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                                if is_binary:
                                    labels_pie = ['Normal', 'Attack']
                                    sizes_pie = [pred_counts.get("Normal", 0), pred_counts.get("Attack", 0)]
                                    colors_pie = ['#66b3ff', '#ff9999']
                                else:
                                    normal_count = pred_counts.get('normal', 0)
                                    attack_count = sum(count for label, count in pred_counts.items() if label != 'normal')
                                    labels_pie = ['Normal', 'Attack (All Types)']
                                    sizes_pie = [normal_count, attack_count]
                                    colors_pie = ['#66b3ff', '#ff9999']

                                valid_indices = [i for i, size in enumerate(sizes_pie) if size > 0]
                                if not valid_indices:
                                    ax_pie.text(0.5, 0.5, 'No data to display', ha='center', va='center')
                                else:
                                    labels_pie_valid = [labels_pie[i] for i in valid_indices]
                                    sizes_pie_valid = [sizes_pie[i] for i in valid_indices]
                                    colors_pie_valid = [colors_pie[i] for i in valid_indices]
                                    ax_pie.pie(sizes_pie_valid, labels=labels_pie_valid, autopct='%1.1f%%', startangle=90, colors=colors_pie_valid,
                                            wedgeprops={'edgecolor': 'white'})
                                    ax_pie.axis('equal')
                                ax_pie.set_title('Prediction Distribution')
                                st.pyplot(fig_pie)

                            st.subheader("📄 Detailed Results")
                            filter_option = st.selectbox(
                                "Filter connections to display:",
                                ["All", "Only Attacks", "Only Normal", "High Confidence Attacks (>90%)", "Low Confidence Predictions (<70%)"]
                            )

                            filtered_results = results_df
                            if is_binary:
                                attack_label_filter = 'Attack'
                                normal_label_filter = 'Normal'
                            else:
                                attack_label_filter = 'normal'
                                normal_label_filter = 'normal'

                            if filter_option == "Only Attacks":
                                if is_binary:
                                    filtered_results = results_df[results_df['prediction_label'] == attack_label_filter]
                                else:
                                    filtered_results = results_df[results_df['prediction_label'] != normal_label_filter]
                            elif filter_option == "Only Normal":
                                filtered_results = results_df[results_df['prediction_label'] == normal_label_filter]
                            elif filter_option == "High Confidence Attacks (>90%)":
                                if is_binary:
                                    filtered_results = results_df[(results_df['prediction_label'] == attack_label_filter) & (results_df['confidence'] > 90)]
                                else:
                                    filtered_results = results_df[(results_df['prediction_label'] != normal_label_filter) & (results_df['confidence'] > 90)]
                            elif filter_option == "Low Confidence Predictions (<70%)":
                                filtered_results = results_df[results_df['confidence'] < 70]

                            st.dataframe(filtered_results)
                            st.markdown(f"Showing **{len(filtered_results)}** connections based on filter.")

                            st.markdown(get_download_link(filtered_results, f"batch_predictions_{filter_option.lower().replace(' ', '_')}.csv", "Download Filtered Results (.csv)"), unsafe_allow_html=True)

                except KeyError as e:
                    st.error(f"Missing Column Error: {e}. This might happen if the batch file structure doesn't match the expected features.")
                    st.error(traceback.format_exc())
                except Exception as e:
                    st.error(f"Error processing batch file: {e}")
                    st.error(traceback.format_exc())

    elif page == "Help":
        st.header("❓ Help & Documentation")

        st.markdown("""
        This page provides guidance on using the Network Intrusion Detection System application.

        ### **Application Workflow**
        1.  **Load Data:** Use the sidebar (`📁 Upload & Load Data`) to upload your network traffic data (CSV/TXT, KDD'99 format, no header) or select the sample data. Click `🔄 Load Data`.
        2.  **Explore Data (Optional):** Navigate to `📊 Data Exploration` to visualize data characteristics, distributions, and correlations.
        3.  **Train Model:** Go to `🧠 Model Training`.
            *   Choose **Binary** (Normal vs. Attack) or **Multiclass** classification.
            *   Adjust **Test set size** and **Number of trees**.
            *   Select **features** for training (or use all).
            *   Click `🚀 Train Model`. Review the performance metrics (Accuracy, Report, Confusion Matrix, Feature Importance).
            *   Optionally, download the trained model (`.pkl`) and scaler (`.pkl`).
        4.  **Predict:**
            *   **Real-time:** Go to `⚡ Real-time Prediction`, fill in the connection details (using the same features the model was trained on), and click `Analyze Connection`.
            *   **Batch:** Go to `📦 Batch Prediction`, upload a file with multiple connections (same format as input, no header), and click `Run Batch Prediction`. View summary and detailed results, and download if needed.
        5.  **Help:** Refer back to this page (`❓ Help & Documentation`) for feature descriptions and FAQs.

        ---
        """)

        st.subheader("Data Format Requirements")
        st.markdown("""
        *   **File Type:** CSV or TXT.
        *   **Header:** Must **not** contain a header row.
        *   **Columns:** Should follow the KDD Cup 1999 dataset structure:
            *   41 features describing the connection.
            *   1 final column containing the attack label (e.g., `normal`, `neptune`, `smurf`). **Crucially, ensure labels do not have trailing dots.** The app cleans them on load, but consistent input is best.
        *   **Encoding:** Categorical features (`protocol_type`, `service`, `flag`) should be text-based (e.g., 'tcp', 'http', 'SF'). The application will encode them during the 'Load Data' step.
        *   **Reference:** [KDD Cup 1999 Data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
        """)

        st.subheader("Feature Descriptions")
        st.markdown("Below are descriptions for the features used in the KDD'99 dataset.")
        col1, col2 = st.columns(2)
        features_list = list(FEATURE_DESCRIPTIONS.items())
        midpoint = len(features_list) // 2 + (len(features_list) % 2)
        with col1:
            for feature, description in features_list[:midpoint]:
                st.markdown(f"**{feature}**: {description}")
        with col2:
            for feature, description in features_list[midpoint:]:
                st.markdown(f"**{feature}**: {description}")
        st.markdown("---")

        st.subheader("Frequently Asked Questions (FAQ)")
        with st.expander("What is the difference between Binary and Multiclass classification?"):
            st.write("""
            *   **Binary Classification:** The model learns to distinguish between only two categories: 'Normal' (0) and 'Attack' (1). It predicts *if* an intrusion occurred but not the specific type. Generally simpler and faster to train.
            *   **Multiclass Classification:** The model learns to distinguish between 'Normal' and multiple specific types of attacks (e.g., 'neptune', 'smurf', 'portsweep'). It predicts *if* an intrusion occurred *and* what type it likely is. More complex but provides more detailed insights.
            """)
        with st.expander("Why is scaling features important?"):
            st.write("""
            Many machine learning algorithms, including Random Forest to some extent (though less sensitive than others like SVM or linear models), perform better when input features are on a similar scale. Features like `src_bytes` can have vastly different ranges than features like `serror_rate` (0-1). `StandardScaler` transforms the data so that each feature has a mean of 0 and a standard deviation of 1. This prevents features with larger values from disproportionately influencing the model. The same scaler fitted during training *must* be used for prediction.
            """)
        with st.expander("How interpretable is the Random Forest model?"):
            st.write("""
            Random Forest models offer a degree of interpretability through **Feature Importance**. This indicates which features the model found most useful *on average* across all trees for making predictions during training. The 'Model Training' page displays the top features. However, understanding *exactly how* these features combine for a *single specific prediction* is more complex than in simpler models like logistic regression. Techniques like SHAP or LIME can provide deeper instance-level explanations but are not implemented in this application.
            """)
        with st.expander("What if my batch prediction file has unknown categories?"):
            st.write("""
            If a categorical feature (like 'service') in your batch upload file contains a value that was not seen during the training phase (when the encoders were fitted), the application will map this unknown category to a special value (-1) before scaling. A warning message will indicate which features had unknown categories and how many were found. This handling prevents errors but might affect prediction accuracy for those specific rows as the model wasn't trained on this "-1" representation. Ideally, your batch data should use the same categories as your training data, or you should retrain the model including these new categories if they are legitimate.
            """)
        with st.expander("Can I use the downloaded model elsewhere?"):
            st.write("""
            Yes. The downloaded `intrusion_detection_model.pkl` file contains the trained Scikit-learn RandomForestClassifier object. The `scaler.pkl` contains the StandardScaler object. You can load these in another Python environment using `joblib.load()` and use them to make predictions on new data, provided the new data is preprocessed (encoded and scaled) in the exact same way (using the loaded scaler and the same feature set/order). Remember to handle categorical features and potential unknown values just like the app does.
            ```python
            import joblib
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import LabelEncoder

            model = joblib.load('intrusion_detection_model.pkl')
            scaler = joblib.load('scaler.pkl')

            print("Note: You need to implement the full preprocessing pipeline (encoding, feature selection, NaN handling) matching the app's logic.")
            ```
            """)

if __name__ == "__main__":
    main()
