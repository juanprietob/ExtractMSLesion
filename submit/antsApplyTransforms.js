const clusterpost = require("clusterpost-lib");
const path = require('path');
const argv = require('minimist')(process.argv.slice(2));
const Converter = require("csvtojson").Converter;
const fs = require('fs');
const _ = require('underscore');
const Promise = require('bluebird');
const spawn = require('child_process').spawn;
const mkdirp = require('mkdirp');

if(!argv["csv"]){
    console.error("Help: ");
    console.error("--csv  Input file for processing");
    process.exit(1);
}

var infile = argv["csv"];
var atlas = argv["atlas"];
var checkExists = argv["checkExists"];

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

var execAntsApplyTransform = function(params){
    return new Promise(function(resolve, reject){
        
        console.log(params);
        var antsApplyTransforms = spawn("antsApplyTransforms", params);

        antsApplyTransforms.stdout.on('data', function(data){
            console.log(String(data));
        });

        antsApplyTransforms.stderr.on('data', function(data){
            console.error(String(data));
        });

        antsApplyTransforms.on('close', function(code){
            if(code){
                reject();
            }else{
                console.log("Done");
                resolve();
            }
        });

    })
    .catch(function(e){
        console.error(e);
    });
}

return getcsvfile(infile)
.then(function(json){

    var allparameters = [];

    if(atlas){

        var patient0 = _.min(json, function(patient){
            return Number(patient.ID);
        });

        _.each(json, function(patient1){

            _.each(json, function(patient2){
                if(patient1.ID != patient2.ID){
                    _.each([{filename: "T2.nii.gz", interpolation: "BSpline[3]"}, {filename: "PD.nii.gz", interpolation: "BSpline[3]"}, {filename: "pvec.nii.gz", interpolation: "NearestNeighbor"}], function(item){
                    //_.each([{filename: "pvec.nii.gz", interpolation: "NearestNeighbor"}], function(item){
                        
                        var params = [];
                        params.push("-d");
                        params.push("3");// 24755597_20111207_PDT2-20311270_20090507_PDT2Affine.txt
                        var patientid0 = String(patient0["ID"]);
                        var patientid1 = String(patient1["ID"]);
                        var patientid2 = String(patient2["ID"]);

                        var date0 = String(patient0["date"]);
                        var date1 = String(patient1["date"]);
                        var date2 = String(patient2["date"]);

                        params.push("-t");
                        params.push(path.join("CLIMB-registration-atlas", patientid0, date0, "MNI152_T1_1mm-" + date0 + "_PD_T2Warp.nii.gz"));
                        params.push("-t");
                        params.push(path.join("CLIMB-registration-atlas", patientid0, date0, "MNI152_T1_1mm-" + date0 + "_PD_T2Affine.txt"));                        

                        if(patientid0 !== patientid1){
                            params.push("-t");
                            params.push(path.join("CLIMB-registration-atlas", patientid0, date0, patientid0 + "_" + date0 + "_PD_T2-" + patientid1 + "_" + date1 + "_PD_T2Warp.nii.gz"));
                            params.push("-t");
                            params.push(path.join("CLIMB-registration-atlas", patientid0, date0, patientid0 + "_" + date0 + "_PD_T2-" + patientid1 + "_" + date1 + "_PD_T2Affine.txt"));
                        }

                        params.push("-t");
                        params.push(path.join("CLIMB-registration-atlas", patientid1, date1, patientid1 + "_" + date1 + "_PD_T2-" + patientid2 + "_" + date2 + "_PD_T2Warp.nii.gz"));
                        params.push("-t");
                        params.push(path.join("CLIMB-registration-atlas", patientid1, date1, patientid1 + "_" + date1 + "_PD_T2-" + patientid2 + "_" + date2 + "_PD_T2Affine.txt"));
                        
                        params.push("-i");
                        params.push(path.join("CLIMB", patientid2, date2, item.filename));
                        params.push("-r");
                        params.push(path.join("CLIMB", "MNI152_T1_1mm.nii.gz"));
                        params.push("-n");
                        params.push(item.interpolation);
                        params.push("-o");
                        try{
                            fs.statSync(path.join("CLIMB-registered-atlas", patientid2, date2));
                        }catch(e){
                            mkdirp.sync(path.join("CLIMB-registered-atlas", patientid2, date2));
                        }
                        params.push(path.join("CLIMB-registered-atlas", patientid2, date2, "MNI152_T1_1mm-" + patientid1 + "_" + date1 + "-" + patientid2 + "_" + date2 + item.filename));
                        if(checkExists){
                            try{
                                var outfile = params[params.length-1];
                                fs.statSync(outfile);
                                console.log("Already generated:", outfile)
                            }catch(e){
                                allparameters.push(params);
                            }
                        }else{
                            allparameters.push(params);
                        }
                    });
                }
            });

        });

    }else{

        _.each(json, function(patient){

            var tp = [];
            _.each(patient, function(t, key){
                if(key.indexOf("field") !== -1){
                    tp.push(String(t));
                }
            });

            _.each([{filename: "T2.nii.gz", interpolation: "BSpline[3]"}, {filename: "PD.nii.gz", interpolation: "BSpline[3]"}, {filename: "pvec.nii.gz", interpolation: "NearestNeighbor"}], function(item){
            //_.each([{filename: "pvec.nii.gz", interpolation: "NearestNeighbor"}], function(item){
                for(var i = 0; i < tp.length ; i++){
                    var params = [];
                    params.push("-d");
                    params.push("3");
                    var patientid = String(patient["ID"]);
                    params.push("-t");
                    params.push(path.join("CLIMB-registration", patientid, "MNI152_T1_1mm-avg_PD_T2-" + tp[0] + "_PD_T2Warp.nii.gz"));
                    params.push("-t");
                    params.push(path.join("CLIMB-registration", patientid, "MNI152_T1_1mm-avg_PD_T2-" + tp[0] + "_PD_T2Affine.txt"));
                    for(var j = 0; j < i; j++){
                        params.push("-t");
                        params.push(path.join("CLIMB-registration", patientid, tp[j] + "_PD_T2-" + tp[j+1] + "_PD_T2Warp.nii.gz"));
                        params.push("-t");
                        params.push(path.join("CLIMB-registration", patientid, tp[j] + "_PD_T2-" + tp[j+1] + "_PD_T2Affine.txt"));
                    }
                    params.push("-i");
                    params.push(path.join("CLIMB", patientid, tp[i], item.filename));
                    params.push("-r");
                    params.push(path.join("CLIMB", "MNI152_T1_1mm.nii.gz"));
                    params.push("-n");
                    params.push(item.interpolation);
                    params.push("-o");
                    try{
                        fs.statSync(path.join("CLIMB-registered-avg", patientid, tp[i]));
                    }catch(e){
                        mkdirp.sync(path.join("CLIMB-registered-avg", patientid, tp[i]));
                    }
                    params.push(path.join("CLIMB-registered-avg", patientid, tp[i], item.filename));
                    if(checkExists){
                        try{
                            var outfile = params[params.length-1];
                            fs.statSync(outfile);
                            console.log("Already generated:", outfile)
                        }catch(e){
                            allparameters.push(params);
                        }
                    }else{
                        allparameters.push(params);
                    }
                }
            });
            
        });
    }

    return Promise.map(allparameters, execAntsApplyTransform,
    {
        concurrency: 4
    });
    
})
.then(console.log);
