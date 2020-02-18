from flask import Flask, request
import requests
import os.path

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional

path = './'

export_file_url = 'https://www.dropbox.com/s/l6i6jlssxlzsdk9/service_desk_oneday.pt?dl=0'
export_file_name = 'service_desk.pt'

data_model_mappings = {}


class Net(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        global data_model_mappings

        super(Net, self).__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        output_width = len(data_model_mappings['Completion_Category'])
        input_width = (self.n_emb + self.n_cont) * 50

        if input_width > 500:
            input_width = 500

        hidden_width = (input_width // 8) + output_width

        input_width = 200
        hidden_width = 70

        print(
            f"First layer dim: ({self.n_emb + self.n_cont}, {input_width}), Labels: {output_width}, Inner: {hidden_width}")

        # self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        # self.lin2 = nn.Linear(200, 70)
        # self.lin3 = nn.Linear(70, 4)
        # self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(self.n_emb + self.n_cont)
        self.bn3 = nn.BatchNorm1d(input_width)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

        self.fc1 = nn.Linear(self.n_emb + self.n_cont, input_width)
        self.fc2 = nn.Linear(input_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, output_width)

    def forward(self, x_cat, x_cont):
        x: list = [e(x_cat[:, i].long()) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
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

checkpoint = torch.load(path + '/' + export_file_name)
# final_epoch = checkpoint['epoch']
# final_loss = checkpoint['loss']
es = checkpoint['embedding_sizes']
td = checkpoint['training_data']
data_model_mappings = checkpoint['data_mappings']

# create network
net = Net(es, 0)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# create optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    label_map = {}

    svc_sub_type = None
    team_name = None
    sla_recid = None

    if request.method == 'POST':  # this block is only entered when the form is submitted
        svc_sub_type = request.form.get('ServiceSubType')
        team_name = request.form.get('team_name')
        sla_recid = request.form.get('SR_SLA_RecID')

    for entry in data_model_mappings.keys():
        label_map[entry] = ''
        for idx, label in data_model_mappings[entry].items():
            label = str(label)

            label_map[label] = idx

            selected = ''
            if label == svc_sub_type or label == team_name or label == sla_recid:
                selected = ' selected'

            label_map[entry] = label_map[entry] + '<option value="' + str(label) + '" ' + selected + '>' + str(label) + '</option>'

    if request.method == 'POST':  # this block is only entered when the form is submitted
        svc_sub_type = label_map[request.form.get('ServiceSubType')]
        team_name = label_map[request.form.get('team_name')]
        sla_recid = label_map[request.form.get('SR_SLA_RecID')]

        outputs = net.forward(torch.tensor([[svc_sub_type, team_name, sla_recid]]), 0)
        _, y_hat = outputs.max(1)

        print(f"Eval: {svc_sub_type, team_name, sla_recid} = {data_model_mappings['Completion_Category'][y_hat.item()]}")

        pred = f"between {2.5 * data_model_mappings['Completion_Category'][y_hat.item()]}" + \
               f" and {2.5 * (data_model_mappings['Completion_Category'][y_hat.item()] + 1)} days"

        return '''
            <html lang="en" itemscope itemtype="https://schema.org/Article">
        <head>
        <link href="https://fonts.googleapis.com/css?family=Lato:100,400,400i,700,900" rel="stylesheet">
        <link rel='stylesheet' id='wp-block-library-css'  href='https://www.edgetg.com/wp-includes/css/dist/block-library/style.min.css?ver=5.2.2' type='text/css' media='all' />
        <link rel='stylesheet' id='bcct_style-css'  href='https://www.edgetg.com/wp-content/plugins/better-click-to-tweet/assets/css/styles.css?ver=3.0' type='text/css' media='all' />
        <link rel='stylesheet' id='main-stylesheet-css'  href='https://www.edgetg.com/wp-content/themes/edge-technology-group/dist/assets/css/app.css' type='text/css' media='all' />
        <link rel='stylesheet' id='addthis_all_pages-css'  href='https://www.edgetg.com/wp-content/plugins/addthis/frontend/build/addthis_wordpress_public.min.css?ver=5.2.2' type='text/css' media='all' />
        </head>
        
                <header class="site-header">
            <div class="grid-container">
                <div class="header-logo"> 
					<a href="https://www.edgetg.com" title=""><img src="https://www.edgetg.com/wp-content/uploads/2018/08/header-logo.svg" alt="" /></a>
				</div>
                <div class="header-menu">
                    <div class="header-top">
                        <ul>
                            <li><a href="https://www.edgetg.com/locations/" title="Locations">Locations</a></li><li>
								<ul class="nav_dropdown">
									<li class="button-dropdown">
										<a href="javascript:void(0)" class="dropdown-toggle"> 
											<img src="https://www.edgetg.com/wp-content/themes/edge-technology-group/dist/assets/images/phone-icon.svg" alt=""> 
										</a>
										<ul class="dropdown-menu"><li>New York: +1 203 742 7800</li><li>San Francisco: +1 415 293 8160</li><li>Austin: +1 203 501 7922</li><li>London: +44 203 535 7800</li><li>Hong Kong: +852 3899 8200</li><li>Singapore: +65 6513 2180</li><li>Sydney: +61 2 9158 8438</li></ul></li></ul></li><li><a href="mailto:info@edgetg.com?subject=Edge Technology Group Website Inquiry" title="">
								<img src="https://www.edgetg.com/wp-content/themes/edge-technology-group/dist/assets/images/envelope-icon.svg" alt="" />
							</a></li>                            <li>
                                <div class="search-wrapper">
                                    <form role="search" method="get" class="search-form" action="https://www.edgetg.com">
                                        <div class="input-group hover-trigger">
                                            <div class="input-group-button hide-for-small-only">
                                                <input type="submit" class="search-submit button" value=""></div>
                                                <input type="search" class="search-field input-group-field" placeholder="Search"
                                                   value="" name="s" title="Search for:">
                                            <div class="input-group-button show-for-small-only">
                                                <input type="submit" class="search-submit button" value="Search"></div>
                                        </div>
                                    </form>
                                </div>
                            </li>
                        </ul>
                    </div>
                    <div class="menu-header-container"><ul id="menu-header" class="enumenu_ul menu"><li id="menu-item-61" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-61"><a href="https://www.edgetg.com/solutions/">Solutions</a>
<ul class="sub-menu">
	<li id="menu-item-518" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-518"><a href="https://www.edgetg.com/solutions/it-support/">Support</a></li>
	<li id="menu-item-62" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-62"><a href="https://www.edgetg.com/solutions/cloud/">Cloud</a></li>
	<li id="menu-item-517" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-517"><a href="https://www.edgetg.com/solutions/cybersecurity/">Cybersecurity</a></li>
	<li id="menu-item-519" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-519"><a href="https://www.edgetg.com/solutions/it-advisory/">Advisory</a></li>
</ul>
</li>
<li id="menu-item-63" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-63"><a href="https://www.edgetg.com/who-we-serve/">Who We Serve</a>
<ul class="sub-menu">
	<li id="menu-item-523" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-523"><a href="https://www.edgetg.com/who-we-serve/hedge-funds/">Hedge Funds</a></li>
	<li id="menu-item-522" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-522"><a href="https://www.edgetg.com/who-we-serve/private-equity/">Private Equity</a></li>
	<li id="menu-item-521" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-521"><a href="https://www.edgetg.com/who-we-serve/family-office/">Family Office</a></li>
	<li id="menu-item-520" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-520"><a href="https://www.edgetg.com/who-we-serve/asset-managers/">Asset Managers</a></li>
</ul>
</li>
<li id="menu-item-1064" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-1064"><a href="https://www.edgetg.com/resources/">Resources</a>
<ul class="sub-menu">
	<li id="menu-item-461" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-461"><a href="https://www.edgetg.com/resources/blog/">Blog</a></li>
</ul>
</li>
<li id="menu-item-66" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-66"><a href="https://www.edgetg.com/careers/">Careers</a>
<ul class="sub-menu">
	<li id="menu-item-534" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-534"><a href="https://www.edgetg.com/careers/jobs/">Jobs</a></li>
</ul>
</li>
<li id="menu-item-67" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-67"><a href="https://www.edgetg.com/about/">About</a>
<ul class="sub-menu">
	<li id="menu-item-525" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-525"><a href="https://www.edgetg.com/about/the-edge-difference/">The Edge Difference</a></li>
	<li id="menu-item-68" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-68"><a href="https://www.edgetg.com/about/executive-team/">Executive Team</a></li>
	<li id="menu-item-524" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-524"><a href="https://www.edgetg.com/about/citizenship/">Citizenship</a></li>
	<li id="menu-item-69" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-69"><a href="https://www.edgetg.com/about/news/">News</a></li>
	<li id="menu-item-627" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-627"><a href="https://www.edgetg.com/locations/">Locations</a></li>
</ul>
</li>
<li id="menu-item-70" class="contact-btn menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-70"><a href="https://www.edgetg.com/contact/">Contact</a>
<ul class="sub-menu">
	<li id="menu-item-762" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-762"><a href="https://www.edgetg.com/contact/client-support/">Client Support</a></li>
</ul>
</li>
</ul></div>                </div>
            </div>
        </header>
        
        <body>
    
        <form method="POST">
        <div class="footer-cta" style="background-image:url('https://www.edgetg.com/wp-content/uploads/2018/09/request-consultation-bg.jpg');">
        <div class="grid-container">
        <div class="main-title"><h2>Service ticket duration prediction:</h4></div>
                <div class='gf_browser_chrome gform_wrapper' id='gform_wrapper_1' ><div id='gf_1' class='gform_anchor' tabindex='-1'></div><form method='post' enctype='multipart/form-data' target='gform_ajax_frame_1' id='gform_1'  action='/#gf_1'>
            <div class="request-consultation-form">
                <div class='gform_body'><ul id='gform_fields_1' class='gform_fields top_label form_sublabel_below description_below'><li id='field_1_1'  class='gfield gfield_contains_required field_sublabel_below field_description_below gfield_visibility_visible' >
                    <label class='gfield_label' for='input_1_1'><p>ServiceSubType</label>
                    <select name="ServiceSubType">
                    ''' + label_map['ticket-v2.servicesubtype'] + '''
                    </select>
                </div>
                <div>
                    <label class='gfield_label' for='input_1_1' ><p>Team Name</label> 
                    <select name="team_name">
                    ''' + label_map['ticket-v2.team_name'] + '''
                    </select>
                </div>
                <div class="menu-item menu-item-type-post_type menu-item-object-page menu-item-523">
                    <label class='gfield_label' for='input_1_1' ><p>SR SLA RecID</label> 
                    <select name="SR_SLA_RecID">
                    ''' + label_map['ticket-v2.sr_status_recid'] + '''
                    </select>
                </div>
                <div>
                    <input id="pred_btn" class='gform_button button' type="submit" value="Submit">
                </div>
                <div>
                    <p>This ticket is expected to take {}
                </div>
                </div>
                </div>

                </div>
          </form>

          </body>
          
        <footer>			
            <div class="grid-container">
                <div class="grid-x">
                    <div class="cell large-12 medium-12 small-12">
                    <div class="footer-menu">
                    <div class="copy-rt">
                        <div class="copy-rt-cont">
                            <ul>
                                <li>&#169;2020 Edge Technology Group</li>
                                <li><a href="https://www.edgetg.com/privacy-policy/" title="Privacy Policy">Privacy Policy</a></li>							</ul>
                        </div>
                        <div class="social-links"> 
                                    <a href="https://www.linkedin.com/company/edge-technology-group/" title="linkedin" target="_blank"><img src="https://www.edgetg.com/wp-content/uploads/2018/08/social-icon-linkedin.svg" alt="" /></a> 
                        </div>					
                    </div>
                </div>
            </div>
        </footer>
        </html>'''.format(pred)
    else:
        return '''
    <html lang="en" itemscope itemtype="https://schema.org/Article">
        <head>
        <link href="https://fonts.googleapis.com/css?family=Lato:100,400,400i,700,900" rel="stylesheet">
        <link rel='stylesheet' id='wp-block-library-css'  href='https://www.edgetg.com/wp-includes/css/dist/block-library/style.min.css?ver=5.2.2' type='text/css' media='all' />
        <link rel='stylesheet' id='bcct_style-css'  href='https://www.edgetg.com/wp-content/plugins/better-click-to-tweet/assets/css/styles.css?ver=3.0' type='text/css' media='all' />
        <link rel='stylesheet' id='main-stylesheet-css'  href='https://www.edgetg.com/wp-content/themes/edge-technology-group/dist/assets/css/app.css' type='text/css' media='all' />
        <link rel='stylesheet' id='addthis_all_pages-css'  href='https://www.edgetg.com/wp-content/plugins/addthis/frontend/build/addthis_wordpress_public.min.css?ver=5.2.2' type='text/css' media='all' />
        </head>
        
                <header class="site-header">
            <div class="grid-container">
                <div class="header-logo"> 
					<a href="https://www.edgetg.com" title=""><img src="https://www.edgetg.com/wp-content/uploads/2018/08/header-logo.svg" alt="" /></a>
				</div>
                <div class="header-menu">
                    <div class="header-top">
                        <ul>
                            <li><a href="https://www.edgetg.com/locations/" title="Locations">Locations</a></li><li>
								<ul class="nav_dropdown">
									<li class="button-dropdown">
										<a href="javascript:void(0)" class="dropdown-toggle"> 
											<img src="https://www.edgetg.com/wp-content/themes/edge-technology-group/dist/assets/images/phone-icon.svg" alt=""> 
										</a>
										<ul class="dropdown-menu"><li>New York: +1 203 742 7800</li><li>San Francisco: +1 415 293 8160</li><li>Austin: +1 203 501 7922</li><li>London: +44 203 535 7800</li><li>Hong Kong: +852 3899 8200</li><li>Singapore: +65 6513 2180</li><li>Sydney: +61 2 9158 8438</li></ul></li></ul></li><li><a href="mailto:info@edgetg.com?subject=Edge Technology Group Website Inquiry" title="">
								<img src="https://www.edgetg.com/wp-content/themes/edge-technology-group/dist/assets/images/envelope-icon.svg" alt="" />
							</a></li>                            <li>
                                <div class="search-wrapper">
                                    <form role="search" method="get" class="search-form" action="https://www.edgetg.com">
                                        <div class="input-group hover-trigger">
                                            <div class="input-group-button hide-for-small-only">
                                                <input type="submit" class="search-submit button" value=""></div>
                                                <input type="search" class="search-field input-group-field" placeholder="Search"
                                                   value="" name="s" title="Search for:">
                                            <div class="input-group-button show-for-small-only">
                                                <input type="submit" class="search-submit button" value="Search"></div>
                                        </div>
                                    </form>
                                </div>
                            </li>
                        </ul>
                    </div>
                    <div class="menu-header-container"><ul id="menu-header" class="enumenu_ul menu"><li id="menu-item-61" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-61"><a href="https://www.edgetg.com/solutions/">Solutions</a>
<ul class="sub-menu">
	<li id="menu-item-518" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-518"><a href="https://www.edgetg.com/solutions/it-support/">Support</a></li>
	<li id="menu-item-62" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-62"><a href="https://www.edgetg.com/solutions/cloud/">Cloud</a></li>
	<li id="menu-item-517" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-517"><a href="https://www.edgetg.com/solutions/cybersecurity/">Cybersecurity</a></li>
	<li id="menu-item-519" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-519"><a href="https://www.edgetg.com/solutions/it-advisory/">Advisory</a></li>
</ul>
</li>
<li id="menu-item-63" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-63"><a href="https://www.edgetg.com/who-we-serve/">Who We Serve</a>
<ul class="sub-menu">
	<li id="menu-item-523" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-523"><a href="https://www.edgetg.com/who-we-serve/hedge-funds/">Hedge Funds</a></li>
	<li id="menu-item-522" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-522"><a href="https://www.edgetg.com/who-we-serve/private-equity/">Private Equity</a></li>
	<li id="menu-item-521" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-521"><a href="https://www.edgetg.com/who-we-serve/family-office/">Family Office</a></li>
	<li id="menu-item-520" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-520"><a href="https://www.edgetg.com/who-we-serve/asset-managers/">Asset Managers</a></li>
</ul>
</li>
<li id="menu-item-1064" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-1064"><a href="https://www.edgetg.com/resources/">Resources</a>
<ul class="sub-menu">
	<li id="menu-item-461" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-461"><a href="https://www.edgetg.com/resources/blog/">Blog</a></li>
</ul>
</li>
<li id="menu-item-66" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-66"><a href="https://www.edgetg.com/careers/">Careers</a>
<ul class="sub-menu">
	<li id="menu-item-534" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-534"><a href="https://www.edgetg.com/careers/jobs/">Jobs</a></li>
</ul>
</li>
<li id="menu-item-67" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-67"><a href="https://www.edgetg.com/about/">About</a>
<ul class="sub-menu">
	<li id="menu-item-525" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-525"><a href="https://www.edgetg.com/about/the-edge-difference/">The Edge Difference</a></li>
	<li id="menu-item-68" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-68"><a href="https://www.edgetg.com/about/executive-team/">Executive Team</a></li>
	<li id="menu-item-524" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-524"><a href="https://www.edgetg.com/about/citizenship/">Citizenship</a></li>
	<li id="menu-item-69" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-69"><a href="https://www.edgetg.com/about/news/">News</a></li>
	<li id="menu-item-627" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-627"><a href="https://www.edgetg.com/locations/">Locations</a></li>
</ul>
</li>
<li id="menu-item-70" class="contact-btn menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-70"><a href="https://www.edgetg.com/contact/">Contact</a>
<ul class="sub-menu">
	<li id="menu-item-762" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-762"><a href="https://www.edgetg.com/contact/client-support/">Client Support</a></li>
</ul>
</li>
</ul></div>                </div>
            </div>
        </header>
        
        <body>
    
        <form method="POST">
        <div class="footer-cta" style="background-image:url('https://www.edgetg.com/wp-content/uploads/2018/09/request-consultation-bg.jpg');">
        <div class="grid-container">
        <div class="main-title"><h2>Service ticket duration prediction:</h4></div>
                <div class='gf_browser_chrome gform_wrapper' id='gform_wrapper_1' ><div id='gf_1' class='gform_anchor' tabindex='-1'></div><form method='post' enctype='multipart/form-data' target='gform_ajax_frame_1' id='gform_1'  action='/#gf_1'>
            <div class="request-consultation-form">
            
                <div class='gform_body'><ul id='gform_fields_1' class='gform_fields top_label form_sublabel_below description_below'><li id='field_1_1'  class='gfield gfield_contains_required field_sublabel_below field_description_below gfield_visibility_visible' >
                    <label class='gfield_label' for='input_1_1'><p>ServiceSubType</label>
                    <select name="ServiceSubType">
                    ''' + label_map['ticket-v2.servicesubtype'] + '''
                    </select>
                </div>
                <div>
                    <label class='gfield_label' for='input_1_1' ><p>Team Name</label> 
                    <select name="team_name">
                    ''' + label_map['ticket-v2.team_name'] + '''
                    </select>
                </div>
                <div class="menu-item menu-item-type-post_type menu-item-object-page menu-item-523">
                    <label class='gfield_label' for='input_1_1' ><p>SR SLA RecID</label> 
                    <select name="SR_SLA_RecID">
                    ''' + label_map['ticket-v2.sr_status_recid'] + '''
                    </select>
                </div>
                <div>
                    <input id="pred_btn" class='gform_button button' type="submit" value="Submit">
                </div>
                </div>
                </div>

                </div>
          </form>

          </body>
          
        <footer>			
            <div class="grid-container">
                <div class="grid-x">
                    <div class="cell large-12 medium-12 small-12">
                    <div class="footer-menu">
                    <div class="copy-rt">
                        <div class="copy-rt-cont">
                            <ul>
                                <li>&#169;2020 Edge Technology Group</li>
                                <li><a href="https://www.edgetg.com/privacy-policy/" title="Privacy Policy">Privacy Policy</a></li>							</ul>
                        </div>
                        <div class="social-links"> 
                                    <a href="https://www.linkedin.com/company/edge-technology-group/" title="linkedin" target="_blank"><img src="https://www.edgetg.com/wp-content/uploads/2018/08/social-icon-linkedin.svg" alt="" /></a> 
                        </div>					
                    </div>
                </div>
            </div>
        </footer>
        </html>
        '''


# app.run(host='0.0.0.0', port=5000, use_reloader=False)
