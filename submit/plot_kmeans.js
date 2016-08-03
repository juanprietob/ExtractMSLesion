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

if(!argv["in"] && !argv["script"]){
    console.error("Help: ");
    console.error("--in pickle file");
    console.error("--out outimg");
    console.error("--script  python script");
    process.exit(1);
}

var input = argv["in"];
var script = argv["script"];
var output = argv["out"];

var inputfiles = [];

inputfiles.push(script);
inputfiles.push(input);


var job = {
    "executable": "python",
    "jobparameters": [
        {
            "flag": "-q",
            "name": "bigmem"
        }
    ],
    "parameters": [
        {
            "flag": "",
            "name": path.basename(script)
        },
        {
            "flag": "--pickle",
            "name": path.basename(input)
        },
        {
        	"flag": "--out",
        	"name": path.basename(output)
        }
    ],
    "inputs": [
        {
            "name": path.basename(script)
        },
        {
            "name": path.basename(input)
        }
    ],
    "outputs": [
        {
            "type": "file",
            "name": path.basename(output)
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
	job.executionserver = res[2].name;
	return clusterpost.createAndSubmitJob(job, inputfiles)
})
.then(function(jobid){
	console.log(input, jobid);
})
.catch(console.error)