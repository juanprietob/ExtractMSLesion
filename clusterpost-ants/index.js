var clusterpost = require("clusterpost-lib");
var path = require('path');
var argv = require('minimist')(process.argv.slice(2));
var stringArgv = require('string-argv');
var Converter=require("csvtojson").Converter;
var fs=require('fs');
var _=require('underscore');
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

var env = process.env.NODE_ENV;
if(!env) throw "Please set NODE_ENV variable.";

var conf = getConfigFile(env, process.cwd());

var remotefs = conf.uri + '/dataprovider-fs/';

var parameters = [];
var inputs = [];


parameters.push({
    "flag": "",
    "name": String(argv._[0])
});

var addJobInputs = function(jobinput){
    var jobin = path.basename(jobinput);

    if(jobinput[0] == "/"){
        jobinput = jobinput.slice(1);//remove the slash at the start
    }

    inputs.push({
        "name": jobin,
        "remote" : {
            "uri" : remotefs + jobinput
        }
    });
}

try{
    var m = argv["m"];

    if(!m){
        console.error("Please check ants documentation for parameter '-m'");
        process.exit(1);
    }

    var parseMparameter = function(m){

        try{
            var marray = m.substring(m.lastIndexOf("[")+1,m.lastIndexOf("]")).split(",");
            var fixedimagepath = marray[0];
            var fixedimagefilename = path.basename(fixedimagepath);
            var movingimagepath = marray[1];
            var movingimagefilename = path.basename(movingimagepath);
        }catch(e){
            console.error("parseParam", argv);
            throw e + " " + m;
        }
        

        if(fixedimagefilename && movingimagefilename){
            if(fixedimagefilename === movingimagefilename){
                fixedimagefilename = "0_" + fixedimagefilename;
                movingimagefilename = "1_" + movingimagefilename;
            }

            marray[0] = fixedimagefilename;
            marray[1] = movingimagefilename;

            var mm = m.substring(0, m.lastIndexOf("[")) + JSON.stringify(marray).replace(/"/g, '');

            parameters.push({
                "flag": "-m",
                "name": mm
            });

            var input = _.find(inputs, function(input){
                if(remotefs + fixedimagepath === input.remote.uri){
                    return true;
                }
                return false;
            });

            if(!input){
                if(fixedimagepath[0] == "/"){
                    fixedimagepath = fixedimagepath.slice(1);//remove the slash at the start
                }
                inputs.push({
                    "name": fixedimagefilename,
                    "remote" : {
                        "uri" : remotefs + fixedimagepath
                    }
                });    
            }

            var input = _.find(inputs, function(input){
                if(remotefs + movingimagepath === input.remote.uri){
                    return true;
                }
                return false;
            });
            
            if(!input){
                if(movingimagepath[0] == "/"){
                    movingimagepath = movingimagepath.slice(1);//remove the slash at the start
                }
                inputs.push({
                    "name": movingimagefilename,
                    "remote" : {
                        "uri" : remotefs + movingimagepath
                    }
                });
            }
            
        }else{
            throw 'Could not parse "-m ' + m + '"';
        }
    }

    if(_.isArray(m)){
        _.each(m, parseMparameter);
    }else{
        parseMparameter(m);
    }

}catch(e){
    console.error(e);
    process.exit(1);    
}

var x = argv["x"];

if(x){
    addJobInputs(x);

    var xx = path.basename(x);

    parameters.push({
        "flag": "-x",
        "name": xx
    });
}

var jobinputs = argv["jobinputs"];

if(jobinputs){
    if(_.isArray(jobinputs)){
        _.each(jobinputs, addJobInputs);
    }else{
        addJobInputs(jobinputs);
    }
}

var outname = argv["o"];
if(!outname){
    console.error("Please check ants documentation for parameter '-o'");
    process.exit(1);
}
//Parse all the rest of parameters that are specific to ANTS
_.each(argv, function(val, key){
    if(key !== "x" && key !== "m" && key !== "_" && key !== "outputdir" && key !== "jobname" && key !== "jobparameters" && key !== "$0" && key !== "useremail" && key !== "executionserver"){
        var flag = "-";

        if(key.length > 1){
            flag += "-";
        }

        parameters.push({
            "flag": flag+key,
            "name": String(val)
        });
    }
});

var jobparameters;
if(argv["jobparameters"]){
    jobparameters = _.compact(_.map(require('minimist')(stringArgv(argv["jobparameters"])), function(param, key){
        if(key !== "_"){
            var flag = "-";

            if(key.length > 1){
                flag += "-";
            }

            return {
                flag: flag + key,
                name: String(param)
            }
        }
    }));
}


var outputdir = argv["outputdir"];
if(!outputdir){
    console.error("Please set output directory '-outputdir <outputdir>'");
    process.exit(1);
}

var jobname = argv["jobname"];

var getcsvfile = function(filename){
  return new Promise(function(resolve, reject){
    var fileStream=fs.createReadStream(filename);
    //new converter instance
    var csvConverter=new Converter({constructResult:true});

    //end_parsed will be emitted once parsing finished
    csvConverter.on("end_parsed",function(jsonObj){
      resolve(jsonObj);
    });

    fileStream.pipe(csvConverter);
  })
}

clusterpost.setClusterPostServer(conf.uri);

var agentoptions = {
	rejectUnauthorized: false
}

clusterpost.setAgentOptions(agentoptions);

var prom;
try{
    var token = fs.readFileSync(".token");
    clusterpost.setUserToken(token);
    prom = Promise.resolve(token);
}catch(e){

    prom = clusterpost.getUsernamePassword(function(user){
        return clusterpost.userLogin(user);
    })
    .then(function(token){
        fs.writeFileSync(".token", token.token);
        return token;
    });
}

var useremail = "juanprietob@gmail.com";

if(argv["useremail"]){
    useremail = argv["useremail"];
}

var executionserver = "localhost";

if(argv["executionserver"]){
    executionserver = argv["executionserver"];
}

prom
.then(function(res){

    var job = {
        "executable": "ANTS",
        "parameters": parameters,
        "inputs": inputs,
        "outputs": [
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
        "userEmail": useremail
    };
    
    job.executionserver = executionserver;
    
    job.inputs = inputs;

    job.outputs.push({
        type: "file",
        name: outname + "Affine.txt"
    });
    job.outputs.push({
        type: "file",
        name: outname + "Warp.nii.gz"
    });
    job.outputs.push({
        type: "file",
        name: outname + "InverseWarp.nii.gz"
    });

    job.outputdir = outputdir;

    if(jobname){
        job.name = jobname;
    }
    
    if(jobparameters){
        job.jobparameters = jobparameters;
    }

    return clusterpost.createDocument(job)
    .then(function(res){
        var jobid = res.id;
        return clusterpost.executeJob(jobid)
        .then(function(res){
            console.log(jobid);
            return res;
        })
        .catch(function(res){
            console.error(res);
            return res;
        });
    });
})
.then(function(){
    process.exit(0);
})
.catch(console.error)
