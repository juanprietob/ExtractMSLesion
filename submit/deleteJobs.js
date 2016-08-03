var clusterpost = require("clusterpost-lib");
var path = require('path');
var argv = require('minimist')(process.argv.slice(2));
var _ = require('underscore');
var Promise = require('bluebird');


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

if(argv["h"] || argv["help"]){
    console.error("Help: ");
    console.error("--j jobid");
    console.error("--token Set the token to avoid login if running in batch mode (optional)");
    process.exit(1);
}

var jobid = argv["j"];

var env = process.env.NODE_ENV;
if(!env) throw "Please set NODE_ENV variable.";

var conf = getConfigFile(env, process.cwd());

clusterpost.setClusterPostServer(conf.uri);

var agentoptions = {
	rejectUnauthorized: false
}

clusterpost.setAgentOptions(agentoptions);

var login;

if(argv["token"]){
    clusterpost.setUserToken(argv["token"]);
    login = Promise.resolve(argv["token"]);
}else{
    login = clusterpost.userLogin(conf.user);
}

login
.then(function(res){
    if(jobid){
        return clusterpost.deleteJob(jobid);
    }else{
        return clusterpost.getJobs("extractMSLesion", "FAIL")
        .then(function(jobs){
            return Promise.map(_.pluck(jobs, "_id"), clusterpost.deleteJob);
        });
    }
	
})
.then(function(res){
	console.log(res);
})
.catch(console.error)