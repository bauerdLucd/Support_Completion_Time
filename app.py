from flask import Flask, request
import requests
import os.path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional

path = ''

export_file_url = 'https://www.dropbox.com/s/j5pi1o2mxzfwtcf/10Oct19_56.pkl?dl=1'
export_file_name = '10Oct19_56.pkl'


class Net(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        global output_labels

        super(Net, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        # length of all embeddings combined
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont

        width = (self.n_emb + self.n_cont) * 50

        if width > 500:
            width = 500

        inner_width = (width // 8) + output_labels

        width = 200
        inner_width = 70

        print(f"First layer dim: ({self.n_emb + self.n_cont}, {width}), Labels: {output_labels}, Inner: {inner_width}")

        # self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        # self.lin2 = nn.Linear(200, 70)
        # self.lin3 = nn.Linear(70, 4)
        # self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(self.n_emb + self.n_cont)
        self.bn3 = nn.BatchNorm1d(width)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

        self.fc1 = nn.Linear(self.n_emb + self.n_cont, width)
        self.fc2 = nn.Linear(width, inner_width)
        self.fc3 = nn.Linear(inner_width, output_labels)

    def forward(self, x_cat, x_cont):
        # print(f"Embeddings: {self.embeddings}")
        # print(f"xcat: {x_cat}")

        x = []
        for i, e in enumerate(self.embeddings):
            x.append(e(x_cat[:, i].long()))

        x = torch.cat(x, 1)
        # x = self.emb_drop(x)
        # x2 = self.bn1(x_cont)
        # x = torch.cat([x, x2], 1)
        # x = torch.cat([x], 1)
        # x = functional.relu(self.lin1(x))
        # x = self.drops(x)
        # x = self.bn2(x)
        # x = functional.relu(self.lin2(x))
        # x = self.drops(x)
        # x = self.bn3(x)
        # x = self.lin3(x)

        x = self.emb_drop(x)
        x = self.bn2(x)
        x = functional.relu(self.fc1(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def down_load_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)


def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        down_load_file(filename, url)
        return True
    return False


download_if_not_exists(export_file_name, export_file_url)

checkpoint = torch.load(path, export_file_name)
# final_epoch = checkpoint['epoch']
# final_loss = checkpoint['loss']
es = checkpoint['embedding_sizes']
td = checkpoint['training_data']

# create network
net = Net(es, None)
net.load_state_dict(torch.load())
net.load_state_dict(checkpoint['model_state_dict'])

# create optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':  # this block is only entered when the form is submitted

        Urgency = request.form.get('Urgency')

        Source = request.form.get('Source')

        ServiceType = request.form.get('ServiceType')

        ServiceSubType = request.form.get('ServiceSubType')

        ServiceSubTypeItem = request.form.get('ServiceSubTypeItem')

        team_name = request.form.get('team_name')

        Territory = request.form.get('Territory')

        Hours_Billable = request.form.get('Hours_Billable')

        Hours_NonBillable = request.form.get('Hours_NonBillable')

        SR_SLA_RecID = request.form.get('SR_SLA_RecID')

        WEEKDAY = request.form.get('WEEKDAY')

        Working_Hours = request.form.get('Working_Hours')

        Scheduled = request.form.get('Scheduled')

        inf_df = pd.DataFrame(columns=['Urgency', 'Source', 'ServiceType', 'ServiceSubType',
                                       'ServiceSubTypeItem', 'team_name', 'Territory', 'Hours_Billable',
                                       'Hours_NonBillable', 'SR_SLA_RecID', 'WEEKDAY', 'Working_Hours', 'Scheduled'])
        inf_df.loc[0] = [Urgency, Source, ServiceType, ServiceSubType, ServiceSubTypeItem, team_name,
                         Territory, Hours_Billable, Hours_NonBillable, SR_SLA_RecID, WEEKDAY, Working_Hours, Scheduled]

        inf_df['Hours_Billable'] = inf_df['Hours_Billable'].astype(float)
        inf_df['Hours_NonBillable'] = inf_df['Hours_NonBillable'].astype(float)
        inf_df['SR_SLA_RecID'] = inf_df['SR_SLA_RecID'].astype(int)
        inf_df['WEEKDAY'] = (inf_df.WEEKDAY == 'True')
        inf_df['Working_Hours'] = (inf_df.Working_Hours == 'True')
        inf_df['Scheduled'] = (inf_df.Scheduled == 'True')

        inf_df.loc[inf_df['ServiceType'] == 'Blank', 'ServiceType'] = np.nan
        inf_df.loc[inf_df['ServiceSubType'] == 'Blank', 'ServiceSubType'] = np.nan
        inf_df.loc[inf_df['ServiceSubTypeItem'] == 'Blank', 'ServiceSubTypeItem'] = np.nan

        inf_row = inf_df.iloc[0]

        #        print(inf_row)
        #        print(inf_df.dtypes)

        pred = prediction = net([0,0,0])
        # pred = learn.predict(inf_row)

        return '''The input Urgency is: {}<br>
                    The input Source is: {}<br>
                    The input ServiceType is: {}<br>
                    The input ServiceSubType is: {}<br>
                    The input ServiceSubTypeItem is: {}<br>
                    The input team_name is: {}<br>
                    The input Territory is: {}<br>
                    The input Hours_Billable is: {}<br>
                    The input Hours_NonBillable is: {}<br>
                    The input SR_SLA_RecID is: {}<br>
                    The input on whether the request is on a WEEKDAY is: {}<br>
                    The input on whether the request is during Working_Hours is: {}<br>
                    The input on whether the request is Scheduled is: {}<br>
                    <h1>The time to completion for the support request is predicted to be: {}</h1>
                    <h3>For reference val1 10min to 1hr, val2 1hr to 1day, val3 <10min, val4 >1day'''.format(Urgency,
                                                                                                             Source,
                                                                                                             ServiceType,
                                                                                                             ServiceSubType,
                                                                                                             ServiceSubTypeItem,
                                                                                                             team_name,
                                                                                                             Territory,
                                                                                                             Hours_Billable,
                                                                                                             Hours_NonBillable,
                                                                                                             SR_SLA_RecID,
                                                                                                             WEEKDAY,
                                                                                                             Working_Hours,
                                                                                                             Scheduled,
                                                                                                             pred)

    return '''<form method="POST">
                  <h1>Predicting how long a support request will take to complete</h1>

                  Select Urgency: <select name="Urgency">
                  <option value="Priority 3 - Medium">Priority 3 - Medium</option>
                  <option value="Break/Fix Support">Break/Fix Support</option>
                  <option value="Scheduled Task">Scheduled Task</option>
                  <option value="Priority 1 - Critical">Priority 1 - Critical</option>
                  </select><br>

                  Select Source: <select name="Source">
                  <option value="Email Connector">Email Connector</option>
                  <option value="Internal">Internal</option>
                  <option value="Activity Capture">Activity Capture</option>
                  <option value="Call">Call</option>
                  </select><br>

                  Select ServiceType: <select name="ServiceType">
                  <option value="Desktop">Desktop</option>
                  <option value="Systems">Systems</option>
                  <option value="Monitoring">Monitoring</option>
                  <option value="Spam Filter">Spam Filter</option>
                  <option value="Security (Approval)">Security (Approval)</option>
                  <option value="Blank">Blank</option>
                  <option value="Network">Network</option>
                  <option value="Telecom">Telecom</option>
                  <option value="Mobile Devices">Mobile Devices</option>
                  <option value="Application">Application</option>
                  <option value="Onsite">Onsite</option>
                  <option value="Facilities/Datacenter/Colo">Facilities/Datacenter/Colo</option>
                  <option value="MUST CHANGE">MUST CHANGE</option>
                  <option value="Home Office">Home Office</option>
                  </select><br>

                  Select ServiceSubType: <select name="ServiceSubType">
                  <option value="Exchange">Exchange</option>
                  <option value="Active Directory">Active Directory</option>
                  <option value="Maintenance">Maintenance</option>
                  <option value="Qualify/Escalate">Qualify/Escalate</option>
                  <option value="Blank">Blank</option>
                  <option value="Service Outage">Service Outage</option>
                  <option value="Restore_old">Restore_old</option>
                  <option value="Application">Application</option>
                  <option value="Remote Desktop">Remote Desktop</option>
                  <option value="File Share">File Share</option>
                  <option value="MS Office">MS Office</option>
                  <option value="Blackberry">Blackberry</option>
                  <option value="Disk Space">Disk Space</option>
                  <option value="Skype">Skype</option>
                  <option value="FTP">FTP</option>
                  <option value="Add Device">Add Device</option>
                  <option value="Backup">Backup</option>
                  <option value="SAN_OLD">SAN_OLD</option>
                  <option value="Apple">Apple</option>
                  <option value="Virus/Spyware">Virus/Spyware</option>
                  <option value="Spam/Informational/Other">Spam/Informational/Other</option>
                  <option value="Iphone">Iphone</option>
                  <option value="Security">Security</option>
                  <option value="Mailbox Administration">Mailbox Administration</option>
                  <option value="Logical">Logical</option>
                  <option value="Bloomberg">Bloomberg</option>
                  <option value="Printer/Copier/Scanner/FAX">Printer/Copier/Scanner/FAX</option>
                  <option value="Hardware">Hardware</option>
                  <option value="Connectivity">Connectivity</option>
                  <option value="DNS">DNS</option>
                  <option value="Outlook">Outlook</option>
                  <option value="zz-Reports">zz-Reports</option>
                  <option value="VPN">VPN</option>
                  <option value="Android">Android</option>
                  <option value="iPad">iPad</option>
                  <option value="MAC">MAC</option>
                  <option value="zz-Adobe">zz-Adobe</option>
                  <option value="Cooling">Cooling</option>
                  <option value="VMWare_old">VMWare_old</option>
                  <option value="DUO">DUO</option>
                  <option value="eSentireSOC">eSentireSOC</option>
                  <option value="Website">Website</option>
                  <option value="Employee Leave">Employee Leave</option>
                  <option value="Physical">Physical</option>
                  <option value="Internet">Internet</option>
                  </select><br>

                  Select ServiceSubTypeItem: <select name="ServiceSubTypeItem">
                  <option value="Password">Password</option>
                  <option value="Patching">Patching</option>
                  <option value="New User">New User</option>
                  <option value="GPO">GPO</option>
                  <option value="Blank">Blank</option>
                  <option value="Excel">Excel</option>
                  <option value="Install">Install</option>
                  <option value="Systems">Systems</option>
                  <option value="Software">Software</option>
                  <option value="Word">Word</option>
                  <option value="Wireless">Wireless</option>
                  <option value="Outlook">Outlook</option>
                  <option value="Failure">Failure</option>
                  <option value="Network">Network</option>
                  <option value="Remote Access\VPN">Remote Access\VPN</option>
                  <option value="Edge">Edge</option>
                  <option value="Office">Office</option>
                  <option value="CrytoLocker">CrytoLocker</option>
                  </select><br>

                  Select team_name: <select name="team_name">
                  <option value="US - Support">US - Support</option>
                  <option value="Systems">Systems</option>
                  <option value="NetEng">NetEng</option>
                  <option value="NYDesktop">NYDesktop</option>
                  <option value="Partners">Partners</option>
                  </select><br>

                  Select Territory: <select name="Territory">
                  <option value="US">US</option>
                  <option value="Weiss Streamline IT">Weiss Streamline IT</option>
                  <option value="Edge UK">Edge UK</option>
                  <option value="TigerSupport">TigerSupport</option>
                  </select><br>

                  Hours_Billable: <input type="number" name="Hours_Billable" step=0.01 min=0 required="required"><br>

                  Hours_NonBillable: <input type="number" name="Hours_NonBillable" step=0.01 min=0 required="required"><br>

                  Select SR_SLA_RecID: <select name="SR_SLA_RecID">
                  <option value="2">2</option>
                  <option value="6">6</option>
                  <option value="8">8</option>
                  </select><br>

                  Select WEEKDAY: <select name="WEEKDAY">
                  <option value="True">True</option>
                  <option value="False">False</option>
                  </select><br>

                  Select Working_Hours: <select name="Working_Hours">
                  <option value="True">True</option>
                  <option value="False">False</option>
                  </select><br>

                  Select Scheduled: <select name="Scheduled">
                  <option value="True">True</option>
                  <option value="False">False</option>
                  </select><br>

                  <input type="submit" value="Submit"><br>
              </form>'''
