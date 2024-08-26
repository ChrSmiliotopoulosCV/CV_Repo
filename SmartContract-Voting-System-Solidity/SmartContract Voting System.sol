// SPDX-License-Identifier: MIT
pragma solidity ^0.5.6;
contract Election {
    address electionAuthority; //electionAuthority is responsible for starting elections - assigning candidates and voters
    uint electionEndTime; //Declaring the uint type electionEndTime variable  
    uint voterPersNum;
    string[] candidates; // Registered candidates
    address[] addressPool; //The array of type address in which the registered voter's addresses are stored
    mapping (string => uint) votes; // Each candidate's ID is mapped in the respective number of votes
    mapping (address => bool) voters; // Registered voters
    mapping (address => bool) hasVoted; // Checks if a registered voter has voted or not
    mapping (address => uint) votersPriorityNum; // Each voter's address is mapped to the priority number of each voter
    
    //Declaration of constructor for the electionAuthority state variable of the contract
    constructor() public{
        electionAuthority = msg.sender; //msg.sender denotes in Solidity the sender of the message (current call)
    }
    
    /*Modifiers can be used to change the behaviour of functions in a declarative way. For example, you can use a modifier 
    to automatically check a condition prior to executing the function. Modifiers are inheritable properties of contracts 
    and may be overridden by derived contracts, but only if they are marked virtual.*/
    
    modifier only_election_authority() {
        if (msg.sender != electionAuthority) revert(); //If used only the electionAuthority may execute a function
        _;
    }
    
    modifier only_registered_voters() {
        if (!voters[msg.sender]) revert(); //If used only register_voter may execute a function
        _;
    }
    
    modifier vote_only_once() {
        if (hasVoted[msg.sender]) revert(); //If used vote function can be executed only once 
        _;
    }
    
    modifier only_during_election_time() {
        if (electionEndTime == 0 || electionEndTime < block.timestamp) revert(); //If used the function is permitted to be executed only_during_election_time
        _;
    }
    
    /*Functions are the executable units of code. Functions are usually defined inside a contract, but they can also be defined outside of contracts. 
    Function Calls can happen internally or externally and have different levels of visibility towards other contracts. Functions accept parameters 
    and return variables to pass parameters and values between them.*/
    
    /*The start_election function calculates the electionEndTime of the contract*/
    
    /*Pay Attention!!! Duration of the election should be set to 43200 in Linux Epoch time. In real life time it is 12 hours or 720 seconds.*/
    
    function start_election(uint duration) public
        only_election_authority
    {
        electionEndTime = block.timestamp + duration;
    }
    
    /*The register_candidate function allows the registration of new candidates for the election contract*/
    function register_newCandidate(string memory id) public
        only_election_authority
    {
        candidates.push(id);
    }
    
    /*The register_newVoter function allows the registration of new voters to take part to the contract of the elections.*/
    function register_newVoter(address addr, uint pNo) public
        only_election_authority
    {
        voters[addr] = true;
        addressPool.push(addr);
        votersPriorityNum[addr] += pNo;
    }
    
    /*The vote function is executed only once for each candidate to calculate votes to a candidate according to the votersPriorityNum.*/
    function vote(string memory id, address addr) public
        only_registered_voters
        vote_only_once
        only_during_election_time
    {
        votes[id] += 1 * votersPriorityNum[addr];
        hasVoted[msg.sender] = true;
    }
    
    /*get_num_candidates function returns the total number of candidates that take part to the elections*/
    function get_num_candidates() public view returns(uint) {
        
        return candidates.length;
    }
    
    /*get_candidate function returns the id and number of votes of each candidate*/
    function get_candidate(uint i) public
        view returns(string memory _candidate, uint _votes)
    {
        _candidate = candidates[i];
        _votes = votes[_candidate];
    }
    
    /*get_candidate function returns the id and number of votes of each candidate*/
    function get_votersPriorityNum(address addr) public
        view returns(address _votersAddress, uint _votersPriorityNum)
    {
        _votersAddress = addr;
        _votersPriorityNum = votersPriorityNum[addr];
    }
}