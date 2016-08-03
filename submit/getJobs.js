var clusterpost = require("clusterpost-lib");
var path = require('path');
var argv = require('minimist')(process.argv.slice(2));
var Promise = require('bluebird');
var _ = require('underscore');


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

if(argv["help"] || argv["h"]){
    console.error("Options: ");
    console.error("--dir  Output directory");
    console.error("--status  one of [DONE, RUN, FAIL, EXIT, UPLOADING, CREATE]");
    console.error("--j job id");
    process.exit(1);
}

var outputdir = "./out";
if(argv["dir"]){
    outputdir = argv["dir"];
}
var status = argv["status"];
var jobid = argv["j"];
var executable;

if(argv["executable"]){
    executable = argv["executable"];
}

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
            return clusterpost.getJobs(executable, "DONE")
            .then(function(jobs){
                return Promise.map(jobs, function(job){
                    return clusterpost.getJobOutputs(job, path.join(outputdir, job._id))
                    .then(function(){
                        return clusterpost.deleteJob(job._id);
                    });
                }, {
                    concurrency: 2
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
            return clusterpost.getJobs(executable, status)
            .then(function(jobs){
                // return Promise.map(jobs, function(job){
                //     return clusterpost.updateJobStatus(job._id)
                //     .then(function(status){
                //         var obj = {};
                //         obj[job._id] = status;
                        
                //         return obj;
                //     });
                // });
                return _.pluck(jobs, "_id");
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