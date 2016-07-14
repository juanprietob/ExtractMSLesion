var clusterpost = require("clusterpost-lib");
var path = require('path');
var argv = require('minimist')(process.argv.slice(2));
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

if(!argv["dir"]){
    console.error("Help: ");
    console.error("--dir  Output directory");
    process.exit(1);
}

var outputdir = argv["dir"];
var status = argv["status"];
var jobid = argv["j"];

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
    if(!status){
        if(!jobid){
            return clusterpost.getJobs("extractMSLesion", "DONE")
            .then(function(jobs){
                return Promise.map(jobs, function(job){
                    return clusterpost.getJobOutputs(job, path.join(outputdir, job._id));
                })
                .then(function(res){
                    return Promise.map(jobs, function(job){
                        return clusterpost.deleteJob(job._id);
                    });
                });
            });
        }else{
            return clusterpost.getDocument(jobid)
            .then(function(job){
                return clusterpost.getJobOutputs(job, path.join(outputdir, jobid))
            })
            .then(function(){
                return clusterpost.deleteJob(jobid);
            });
        }
    }else{
        if(!jobid){
            return clusterpost.getJobs("extractMSLesion", "RUN")
            .then(function(jobs){
                return Promise.map(jobs, function(job){
                    return clusterpost.updateJobStatus(job._id);
                });
            });
        }else{
            return clusterpost.updateJobStatus(jobid);
        }
    }
    
})
.then(function(res){
    console.log(res);
})
.catch(console.error)