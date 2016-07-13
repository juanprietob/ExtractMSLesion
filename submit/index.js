var clusterpost = require("clusterpost-lib");
var path = require('path');
var argv = require('minimist')(process.argv.slice(2));


const getConfigFile = function (env, base_directory) {
  try {
    // Try to load the user's personal configuration file
    var conf = path.join(base_directory, 'conf.my.' + env + '.json');
    return require(conf);
  } catch (e) {
    // Else, read the default configuration file
    return require(base_directory + '/conf.' + env + '.json');
  }
};



var inputfiles = [];

inputfiles.push(argv["img1"]);
inputfiles.push(argv["img2"]);
inputfiles.push(argv["labelImg"]);

var status = argv["status"];

var kill = argv["kill"];

var jobdelete = argv["delete"];


var job = {
    "executable": "extractMSLesion",
    "parameters": [
        {
            "flag": "--img",
            "name": "PD.nii.gz"
        },
        {
            "flag": "--img",
            "name": "T2.nii.gz"
        },
        {
        	"flag": "-labelValue",
        	"name": "6"
        },
        {
            "flag": "--labelImg",
            "name": "pvec.nii.gz"
        },
        {
        	"flag": "--outDir",
            "name": "./"	
        }
    ],
    "inputs": [
        {
            "name": "PD.nii.gz"
        },
        {
            "name": "T2.nii.gz"
        },
        {
            "name": "pvec.nii.gz"
        }
    ],
    "outputs": [
        {
            "type": "tar.gz",
            "name": "./"
        },
        {
            "type": "file",
            "name": "stdout.out"
        },
        {
            "type": "file",
            "name": "stderr.err"
        }
    ],
    "type": "job",
    "userEmail": "juanprietob@gmail.com"
};

var env = process.env.NODE_ENV;
if(!env) throw "Please set NODE_ENV variable.";

var conf = getConfigFile(env, process.cwd());


clusterpost.setClusterPostServer(conf.uri);

var agentoptions = {
	rejectUnauthorized: false
}

clusterpost.setAgentOptions(agentoptions);

clusterpost.userLogin(conf.user)
.then(function(res){
	return clusterpost.getExecutionServers();
})
.then(function(res){
	job.executionserver = res[0].name;
	return clusterpost.createAndSubmitJob(job, inputfiles)
})
.then(function(jobid){
	console.log(jobid);
})